import CryptoKit
import Darwin
import Foundation

/// Describes compiled code only. Pixel values, adapter factors and optimizer
/// state are runtime feeds and must never be part of a cached package.
struct NativeGraphCodeIdentity: Codable, Equatable {
    struct Tensor: Codable, Equatable {
        let name: String, dtype: String
        let shape: [Int]
    }
    struct Layer: Codable, Equatable {
        let name: String
        let weightShape: [Int], rank: Int
        let alphaBits: UInt32
    }
    var schema = 1
    var executableSHA256: String
    var osVersion: String, osBuild: String
    var metalName: String
    var metalRegistryID: UInt64, maximumTrainingBytes: UInt64
    var baseSHA256: String
    var width: Int, height: Int
    var target: String
    var architecture: [Int]
    var tensors: [Tensor]
    var layers: [Layer]

    var fingerprint: String {
        var ordered = self
        ordered.tensors.sort { $0.name < $1.name }
        ordered.layers.sort { $0.name < $1.name }
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        // This identity contains only finite integer and string values.
        return NativeGraphCodeCache.digest(try! encoder.encode(ordered))
    }
    static let executableSHA256: String? = {
        guard let url = Bundle.main.executableURL else { return nil }
        return try? NativeGraphCodeCache.fileDigest(url)
    }()
    static var osBuild: String {
        var length = 0
        guard sysctlbyname("kern.osversion", nil, &length, nil, 0) == 0, length > 0 else { return "unknown" }
        var bytes = [CChar](repeating: 0, count: length)
        guard sysctlbyname("kern.osversion", &bytes, &length, nil, 0) == 0 else { return "unknown" }
        return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}

/// Immutable compiled packages shared by equivalent Programs and processes.
/// One shared lease pins a complete identity while a Program can use it.
/// Publication and eviction take a cross-process lock, and eviction requires an
/// exclusive identity lease. Cache failures always leave compilation available.
final class NativeGraphCodeCache {
    struct Package {
        let url: URL
        let metadata: Data
    }
    private struct FileRecord: Codable, Equatable {
        let path: String
        let bytes: UInt64
        let sha256: String
    }
    private struct Record: Codable {
        let key: String, generation: String
        let metadata: Data
        let metadataSHA256: String
        let files: [FileRecord]
    }
    private struct GenerationReference: Decodable {
        let generation: String
    }
    private struct CatalogEntry: Codable {
        var bytes: UInt64
        var lastUsed: TimeInterval
    }
    private let root: URL, namespaces: URL, leases: URL, directory: URL
    private let identity: String, maximumBytes: UInt64
    private let globalFD: Int32
    private var leaseFD: Int32 = -1
    private let processLock = NSLock()

    static var defaultRoot: URL? {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first?
            .appendingPathComponent("Texture Studio/NativeTrainingPrograms/v1", isDirectory: true)
    }
    init(root: URL, identity: String, maximumBytes: UInt64 = 1 << 30) throws {
        guard Self.isFingerprint(identity), maximumBytes > 0 else { throw CocoaError(.fileReadCorruptFile) }
        self.root = root; self.identity = identity; self.maximumBytes = maximumBytes
        namespaces = root.appendingPathComponent("programs", isDirectory: true)
        leases = root.appendingPathComponent("leases", isDirectory: true)
        directory = namespaces.appendingPathComponent(identity, isDirectory: true)
        try FileManager.default.createDirectory(at: namespaces, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: leases, withIntermediateDirectories: true)
        globalFD = Darwin.open(root.appendingPathComponent(".lock").path, O_CREAT | O_RDWR | O_NOFOLLOW, S_IRUSR | S_IWUSR)
        guard globalFD >= 0 else { throw CocoaError(.fileWriteUnknown) }
        do {
            try locked {
                leaseFD = Darwin.open(leases.appendingPathComponent(identity + ".lock").path, O_CREAT | O_RDWR | O_NOFOLLOW, S_IRUSR | S_IWUSR)
                guard leaseFD >= 0 else { throw CocoaError(.fileReadUnknown) }
                let canRecover = flock(leaseFD, LOCK_EX | LOCK_NB) == 0
                if !canRecover {
                    guard flock(leaseFD, LOCK_SH | LOCK_NB) == 0 else { throw CocoaError(.fileReadUnknown) }
                }
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                if canRecover {
                    // No other Program can be using this identity. Remove only
                    // immutable generations no longer named by a stage index;
                    // interrupted publication and replacements otherwise keep
                    // occupying quota across every subsequent training job.
                    try recoverUnreferencedGenerations()
                    guard flock(leaseFD, LOCK_SH | LOCK_NB) == 0 else { throw CocoaError(.fileReadUnknown) }
                }
                // Reconcile crash leftovers once per Program. Publication
                // reserves quota before copying, so intervening users can only
                // overestimate, never omit an uncompleted write's allocation.
                var catalog = try rebuildCatalog()
                catalog[identity]?.lastUsed = Date.timeIntervalSinceReferenceDate
                trim(&catalog, adding: 0)
                try saveCatalog(catalog)
            }
        } catch {
            if leaseFD >= 0 { flock(leaseFD, LOCK_UN); Darwin.close(leaseFD); leaseFD = -1 }
            Darwin.close(globalFD)
            throw error
        }
    }
    deinit {
        if leaseFD >= 0 { flock(leaseFD, LOCK_UN); Darwin.close(leaseFD) }
        Darwin.close(globalFD)
    }
    func package(for key: String) -> Package? {
        try? locked {
            let result = try readPackage(key)
            if result != nil { try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: directory.path) }
            return result
        }
    }
    /// The source remains available for the temporary-cache fallback. Each
    /// generation is immutable; replacing metadata never overwrites live code.
    func store(package source: URL, metadata: Data, key: String) -> Package? {
        guard let files = try? Self.inventory(source), !files.isEmpty else { return nil }
        let packageBytes = files.reduce(UInt64(0)) { $0 + $1.bytes }
        guard packageBytes > 0, packageBytes <= maximumBytes else { return nil }
        return try? locked {
            if let existing = try? readPackage(key), existing.metadata == metadata { return existing }
            let generation = UUID().uuidString
            let record = Record(key: key, generation: generation, metadata: metadata,
                metadataSHA256: Self.digest(metadata), files: files)
            let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
            let bytes = try encoder.encode(record)
            let cost = packageBytes + UInt64(bytes.count)
            var catalog: [String: CatalogEntry]
            if let saved = try? readCatalog() { catalog = saved }
            else { catalog = try rebuildCatalog() }
            trim(&catalog, adding: cost)
            let total = catalog.values.reduce(UInt64(0)) { $0 + $1.bytes }
            guard cost <= maximumBytes, total <= maximumBytes - cost else { throw CocoaError(.fileWriteOutOfSpace) }
            let previous = catalog[identity]?.bytes ?? 0
            catalog[identity] = CatalogEntry(bytes: previous + cost, lastUsed: Date.timeIntervalSinceReferenceDate)
            // Reserve first. A crash or failed copy leaves conservative quota
            // until the next Program reconciles actual on-disk bytes.
            try saveCatalog(catalog)
            let destination = directory.appendingPathComponent(generation + ".mpsgraphpackage", isDirectory: true)
            do {
                try FileManager.default.copyItem(at: source, to: destination)
                try bytes.write(to: indexURL(key), options: .atomic)
                try? FileManager.default.setAttributes([.modificationDate: Date()], ofItemAtPath: directory.path)
                return Package(url: destination, metadata: metadata)
            } catch {
                try? FileManager.default.removeItem(at: destination)
                throw error
            }
        }
    }
    private func readPackage(_ key: String) throws -> Package? {
        guard FileManager.default.fileExists(atPath: indexURL(key).path) else { return nil }
        let record = try JSONDecoder().decode(Record.self, from: Data(contentsOf: indexURL(key)))
        guard record.key == key, UUID(uuidString: record.generation) != nil, !record.files.isEmpty,
              record.metadataSHA256 == Self.digest(record.metadata) else { return nil }
        let package = directory.appendingPathComponent(record.generation + ".mpsgraphpackage", isDirectory: true)
        // Verify every code byte and reject missing, extra or changed files
        // before passing a package to the framework's nonthrowing initializer.
        guard try Self.inventory(package) == record.files else { return nil }
        return Package(url: package, metadata: record.metadata)
    }
    private func indexURL(_ key: String) -> URL { directory.appendingPathComponent(Self.digest(Data(key.utf8)) + ".json") }
    private func locked<T>(_ body: () throws -> T) throws -> T {
        // Compiled-code reuse is optional. A contending publisher must never
        // stall model execution or Abort while holding a process/file lock.
        guard processLock.try() else { throw CocoaError(.fileWriteUnknown) }
        defer { processLock.unlock() }
        guard flock(globalFD, LOCK_EX | LOCK_NB) == 0 else { throw CocoaError(.fileWriteUnknown) }
        defer { flock(globalFD, LOCK_UN) }
        return try body()
    }
    private func trim(_ catalog: inout [String: CatalogEntry], adding bytes: UInt64) {
        var total = catalog.values.reduce(UInt64(0)) { $0 + $1.bytes }
        for (name, item) in catalog.sorted(by: { $0.value.lastUsed < $1.value.lastUsed }) {
            if bytes <= maximumBytes, total <= maximumBytes - bytes { break }
            let fd = Darwin.open(leases.appendingPathComponent(name + ".lock").path, O_CREAT | O_RDWR | O_NOFOLLOW, S_IRUSR | S_IWUSR)
            guard fd >= 0 else { continue }
            defer { Darwin.close(fd) }
            guard flock(fd, LOCK_EX | LOCK_NB) == 0 else { continue }
            defer { flock(fd, LOCK_UN) }
            do {
                try FileManager.default.removeItem(at: namespaces.appendingPathComponent(name, isDirectory: true))
                catalog.removeValue(forKey: name); total -= item.bytes
            } catch { /* In-use or inaccessible code remains conservatively counted. */ }
        }
    }
    private func readCatalog() throws -> [String: CatalogEntry] {
        let result = try JSONDecoder().decode([String: CatalogEntry].self, from: Data(contentsOf: root.appendingPathComponent("catalog.json")))
        guard result.keys.allSatisfy(Self.isFingerprint),
              result.values.allSatisfy({ $0.bytes <= max(maximumBytes, 1 << 30) && $0.lastUsed.isFinite }) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        var total: UInt64 = 0
        for entry in result.values {
            let addition = total.addingReportingOverflow(entry.bytes)
            guard !addition.overflow else { throw CocoaError(.fileReadCorruptFile) }
            total = addition.partialValue
        }
        return result
    }
    private func saveCatalog(_ catalog: [String: CatalogEntry]) throws {
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        try encoder.encode(catalog).write(to: root.appendingPathComponent("catalog.json"), options: .atomic)
    }
    private func recoverUnreferencedGenerations() throws {
        let urls = try FileManager.default.contentsOfDirectory(at: directory,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey])
        var referenced = Set<UUID>()
        for url in urls where url.pathExtension == "json" && Self.isFingerprint(url.deletingPathExtension().lastPathComponent) {
            let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
            guard values.isRegularFile == true, values.isSymbolicLink != true else { continue }
            // Preserve a valid generation reference even when some other
            // metadata field is damaged and the loader will reject the record.
            if let data = try? Data(contentsOf: url),
               let record = try? JSONDecoder().decode(GenerationReference.self, from: data),
               let generation = UUID(uuidString: record.generation) { referenced.insert(generation) }
        }
        for url in urls where url.pathExtension == "mpsgraphpackage" {
            guard let generation = UUID(uuidString: url.deletingPathExtension().lastPathComponent),
                  !referenced.contains(generation) else { continue }
            let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard values.isDirectory == true, values.isSymbolicLink != true else { continue }
            try FileManager.default.removeItem(at: url)
        }
    }
    private func rebuildCatalog() throws -> [String: CatalogEntry] {
        var result: [String: CatalogEntry] = [:]
        for url in try FileManager.default.contentsOfDirectory(at: namespaces, includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey, .contentModificationDateKey]) {
            let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey, .contentModificationDateKey])
            guard Self.isFingerprint(url.lastPathComponent), values.isDirectory == true, values.isSymbolicLink != true else { continue }
            var total: UInt64 = 0
            guard let items = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey]) else { continue }
            for case let file as URL in items {
                let item = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
                if item.isRegularFile == true, item.isSymbolicLink != true { total += UInt64(max(0, item.fileSize ?? 0)) }
            }
            result[url.lastPathComponent] = CatalogEntry(bytes: total,
                lastUsed: (values.contentModificationDate ?? .distantPast).timeIntervalSinceReferenceDate)
        }
        return result
    }
    private static func inventory(_ directory: URL) throws -> [FileRecord] {
        let rootValues = try directory.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
        guard rootValues.isDirectory == true, rootValues.isSymbolicLink != true else { throw CocoaError(.fileReadCorruptFile) }
        // DirectoryEnumerator expands aliases such as /var to /private/var,
        // while URL.resolvingSymlinksInPath can preserve /var on macOS. Use the
        // filesystem's canonical root for both enumeration and relative paths.
        var canonical = [CChar](repeating: 0, count: Int(PATH_MAX))
        guard directory.path.withCString({ realpath($0, &canonical) }) != nil else { throw CocoaError(.fileReadCorruptFile) }
        let path = String(decoding: canonical.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
        let canonicalDirectory = URL(fileURLWithPath: path, isDirectory: true)
        guard let items = FileManager.default.enumerator(at: canonicalDirectory, includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey, .fileSizeKey]) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        var files: [FileRecord] = []
        let prefix = canonicalDirectory.path + "/"
        for case let file as URL in items {
            let values = try file.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey, .isSymbolicLinkKey, .fileSizeKey])
            guard values.isSymbolicLink != true else { throw CocoaError(.fileReadCorruptFile) }
            if values.isDirectory == true { continue }
            guard values.isRegularFile == true, file.path.hasPrefix(prefix) else { throw CocoaError(.fileReadCorruptFile) }
            files.append(FileRecord(path: String(file.path.dropFirst(prefix.count)), bytes: UInt64(max(0, values.fileSize ?? 0)), sha256: try fileDigest(file)))
        }
        return files.sorted { $0.path < $1.path }
    }
    private static func isFingerprint(_ value: String) -> Bool { value.count == 64 && value.allSatisfy { "0123456789abcdef".contains($0) } }
    static func digest(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
    static func fileDigest(_ url: URL) throws -> String {
        let file = try FileHandle(forReadingFrom: url); defer { try? file.close() }
        var digest = SHA256()
        while let chunk = try file.read(upToCount: 1_048_576), !chunk.isEmpty { digest.update(data: chunk) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

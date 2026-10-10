import Darwin
import Foundation
import XCTest
@testable import TextureStudio

final class NativeGraphCodeCacheTests: XCTestCase {
    private func identity() -> NativeGraphCodeIdentity {
        NativeGraphCodeIdentity(executableSHA256: "engine", osVersion: "27.2", osBuild: "build",
            metalName: "GPU", metalRegistryID: 7, maximumTrainingBytes: 40 << 30,
            baseSHA256: "base", width: 1024, height: 1024, target: "height", architecture: [4, 1, 2, 2, 2, 1, 2, 2],
            tensors: [.init(name: "z", dtype: "F32", shape: [4, 4]), .init(name: "a", dtype: "I64", shape: [64, 64])],
            layers: [.init(name: "z", weightShape: [4, 4], rank: 2, alphaBits: Float(16).bitPattern),
                .init(name: "a", weightShape: [4, 4, 1, 1], rank: 2, alphaBits: Float(16).bitPattern)])
    }
    func testIdentityIsOrderIndependentAndInvalidatesEveryCodeContract() {
        let original = identity()
        var reordered = original; reordered.tensors.reverse(); reordered.layers.reverse()
        XCTAssertEqual(reordered.fingerprint, original.fingerprint)
        let changes: [(inout NativeGraphCodeIdentity) -> Void] = [
            { $0.schema += 1 }, { $0.executableSHA256 = "other engine" }, { $0.osVersion = "other OS" },
            { $0.osBuild = "other build" }, { $0.metalName = "other GPU" }, { $0.metalRegistryID += 1 },
            { $0.maximumTrainingBytes += 1 }, { $0.baseSHA256 = "other base" }, { $0.width += 64 },
            { $0.height += 64 }, { $0.target = "normal" }, { $0.architecture[0] += 1 },
            { $0.tensors[0] = .init(name: "other", dtype: "F32", shape: [4, 4]) },
            { $0.tensors[0] = .init(name: "z", dtype: "I32", shape: [4, 4]) },
            { $0.tensors[0] = .init(name: "z", dtype: "F32", shape: [2, 8]) },
            { $0.layers[0] = .init(name: "other", weightShape: [4, 4], rank: 2, alphaBits: Float(16).bitPattern) },
            { $0.layers[0] = .init(name: "z", weightShape: [2, 8], rank: 2, alphaBits: Float(16).bitPattern) },
            { $0.layers[0] = .init(name: "z", weightShape: [4, 4], rank: 3, alphaBits: Float(16).bitPattern) },
            { $0.layers[0] = .init(name: "z", weightShape: [4, 4], rank: 2, alphaBits: Float(8).bitPattern) }
        ]
        for change in changes {
            var changed = original; change(&changed)
            XCTAssertNotEqual(changed.fingerprint, original.fingerprint)
        }
    }
    func testPackagesSurviveNewProgramsAndCopiesRemainImmutable() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try sourcePackage(root, count: 16)
        // Exercise /var explicitly even when Foundation supplies /private/var
        // as the temporary URL. DirectoryEnumerator expands this parent alias.
        let aliasPath = source.path.hasPrefix("/private/var/") ? String(source.path.dropFirst("/private".count)) : source.path
        let aliasedSource = URL(fileURLWithPath: aliasPath, isDirectory: true)
        let cacheRoot = root.appendingPathComponent("cache", isDirectory: true)
        let fingerprint = identity().fingerprint
        let metadata = Data("shape and feed metadata".utf8)
        let package = try autoreleasepool { () throws -> NativeGraphCodeCache.Package in
            let first = try NativeGraphCodeCache(root: cacheRoot, identity: fingerprint)
            return try XCTUnwrap(first.store(package: aliasedSource, metadata: metadata, key: "forward-1"))
        }
        try Data(repeating: 9, count: 16).write(to: source.appendingPathComponent("code.bin"), options: .atomic)
        let second = try NativeGraphCodeCache(root: cacheRoot, identity: fingerprint)
        let loaded = try XCTUnwrap(second.package(for: "forward-1"))
        XCTAssertEqual(loaded.url, package.url); XCTAssertEqual(loaded.metadata, metadata)
        XCTAssertEqual(try Data(contentsOf: loaded.url.appendingPathComponent("code.bin")), Data(repeating: 3, count: 16))
        var different = identity(); different.executableSHA256 = "new executable"
        let changed = try NativeGraphCodeCache(root: cacheRoot, identity: different.fingerprint)
        XCTAssertNil(changed.package(for: "forward-1"))
    }
    func testCorruptMissingAndIncompleteRecordsBecomeMissesAndCanBeReplaced() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try sourcePackage(root, count: 32)
        let cacheRoot = root.appendingPathComponent("cache", isDirectory: true)
        let fingerprint = identity().fingerprint
        let cache = try NativeGraphCodeCache(root: cacheRoot, identity: fingerprint)
        let key = "reverse-1", metadata = Data("metadata".utf8)
        let first = try XCTUnwrap(cache.store(package: source, metadata: metadata, key: key))
        try Data(repeating: 8, count: 32).write(to: first.url.appendingPathComponent("code.bin"), options: .atomic)
        XCTAssertNil(cache.package(for: key), "Changed compiled bytes must never reach the framework loader.")
        let replacement = try XCTUnwrap(cache.store(package: source, metadata: metadata, key: key))
        XCTAssertNotEqual(replacement.url, first.url)
        XCTAssertTrue(FileManager.default.fileExists(atPath: first.url.path), "A replaced generation may still be in use.")
        XCTAssertNotNil(cache.package(for: key))
        let index = cacheRoot.appendingPathComponent("programs/" + fingerprint)
            .appendingPathComponent(NativeGraphCodeCache.digest(Data(key.utf8)) + ".json")
        var record = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: index)) as? [String: Any])
        record["metadata"] = Data("wrong feed metadata".utf8).base64EncodedString()
        try JSONSerialization.data(withJSONObject: record).write(to: index, options: .atomic)
        XCTAssertNil(cache.package(for: key), "Metadata has its own integrity digest.")
        _ = try XCTUnwrap(cache.store(package: source, metadata: metadata, key: key))
        try Data("unfinished record".utf8).write(to: index, options: .atomic)
        XCTAssertNil(cache.package(for: key))
        let repaired = try XCTUnwrap(cache.store(package: source, metadata: metadata, key: key))
        try FileManager.default.removeItem(at: repaired.url.appendingPathComponent("code.bin"))
        XCTAssertNil(cache.package(for: key))
        try Data("bad catalog".utf8).write(to: cacheRoot.appendingPathComponent("catalog.json"), options: .atomic)
        XCTAssertNotNil(cache.store(package: source, metadata: metadata, key: key), "A corrupt quota catalog is rebuilt.")
    }
    func testEvictionHonorsEverySimultaneousProgramLeaseAndTheGlobalQuota() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try sourcePackage(root, count: 2048)
        let cacheRoot = root.appendingPathComponent("cache", isDirectory: true)
        let firstIdentity = identity().fingerprint
        var first: NativeGraphCodeCache? = try NativeGraphCodeCache(root: cacheRoot, identity: firstIdentity, maximumBytes: 4096)
        let package = try XCTUnwrap(first!.store(package: source, metadata: Data("feeds".utf8), key: "stage"))
        var simultaneous: NativeGraphCodeCache? = try NativeGraphCodeCache(root: cacheRoot, identity: firstIdentity, maximumBytes: 4096)
        var changed = identity(); changed.target = "roughness"
        let second = try NativeGraphCodeCache(root: cacheRoot, identity: changed.fingerprint, maximumBytes: 4096)
        XCTAssertNil(second.store(package: source, metadata: Data("feeds".utf8), key: "stage"))
        XCTAssertNotNil(first!.package(for: "stage"))
        first = nil
        XCTAssertNil(second.store(package: source, metadata: Data("feeds".utf8), key: "stage"))
        XCTAssertNotNil(simultaneous!.package(for: "stage"))
        simultaneous = nil
        XCTAssertNotNil(second.store(package: source, metadata: Data("feeds".utf8), key: "stage"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: package.url.path), "Only an unleased identity can be evicted.")
        let programs = cacheRoot.appendingPathComponent("programs", isDirectory: true)
        let sizes = try FileManager.default.enumerator(at: programs, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey])!.allObjects
            .compactMap { $0 as? URL }.map { try $0.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]) }
        XCTAssertLessThanOrEqual(sizes.filter { $0.isRegularFile == true }.reduce(0) { $0 + ($1.fileSize ?? 0) }, 4096)
    }
    func testRecoveryReclaimsCrashOrphansAndReplacedGenerationsOnlyAfterEveryLeaseEnds() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try sourcePackage(root, count: 1024)
        let cacheRoot = root.appendingPathComponent("cache", isDirectory: true)
        let fingerprint = identity().fingerprint
        var first: NativeGraphCodeCache? = try NativeGraphCodeCache(root: cacheRoot, identity: fingerprint, maximumBytes: 4096)
        let replaced = try XCTUnwrap(first!.store(package: source, metadata: Data("original".utf8), key: "stage"))
        let currentMetadata = Data("replacement".utf8)
        let current = try XCTUnwrap(first!.store(package: source, metadata: currentMetadata, key: "stage"))
        XCTAssertNotEqual(replaced.url, current.url)
        let directory = current.url.deletingLastPathComponent()
        let index = directory.appendingPathComponent(NativeGraphCodeCache.digest(Data("stage".utf8)) + ".json")
        let indexedBytes = try Data(contentsOf: index)
        // Simulate a process dying after it copied compiled code but before it
        // atomically published the stage index naming that generation.
        let orphan = directory.appendingPathComponent(UUID().uuidString + ".mpsgraphpackage", isDirectory: true)
        try FileManager.default.copyItem(at: source, to: orphan)

        let started = ProcessInfo.processInfo.systemUptime
        var simultaneous: NativeGraphCodeCache? = try NativeGraphCodeCache(root: cacheRoot, identity: fingerprint, maximumBytes: 4096)
        XCTAssertLessThan(ProcessInfo.processInfo.systemUptime - started, 1, "A shared lease must not wait for an active Program.")
        XCTAssertTrue(FileManager.default.fileExists(atPath: replaced.url.path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: orphan.path))
        XCTAssertEqual(simultaneous!.package(for: "stage")?.url, current.url)
        XCTAssertNil(simultaneous!.store(package: source, metadata: Data("next".utf8), key: "next-stage"),
            "Unreclaimed generations exhaust the quota while an active Program may still use them.")
        first = nil
        try autoreleasepool {
            let stillLeased = try NativeGraphCodeCache(root: cacheRoot, identity: fingerprint, maximumBytes: 4096)
            XCTAssertEqual(stillLeased.package(for: "stage")?.metadata, currentMetadata)
            XCTAssertTrue(FileManager.default.fileExists(atPath: replaced.url.path), "The remaining shared lease still protects old code.")
            XCTAssertTrue(FileManager.default.fileExists(atPath: orphan.path))
        }
        simultaneous = nil

        let recovered = try NativeGraphCodeCache(root: cacheRoot, identity: fingerprint, maximumBytes: 4096)
        XCTAssertFalse(FileManager.default.fileExists(atPath: replaced.url.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: orphan.path))
        XCTAssertEqual(try Data(contentsOf: index), indexedBytes, "Recovery must preserve the immutable indexed generation.")
        let loaded = try XCTUnwrap(recovered.package(for: "stage"))
        XCTAssertEqual(loaded.url, current.url); XCTAssertEqual(loaded.metadata, currentMetadata)
        XCTAssertEqual(try Data(contentsOf: loaded.url.appendingPathComponent("code.bin")), Data(repeating: 3, count: 1024))
        XCTAssertNotNil(recovered.store(package: source, metadata: Data("next".utf8), key: "next-stage"),
            "Recovery must restore enough quota for new compiled stages in the same identity.")
        let sizes = try FileManager.default.enumerator(at: cacheRoot.appendingPathComponent("programs"),
            includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey])!.allObjects
            .compactMap { $0 as? URL }.map { try $0.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey]) }
        XCTAssertLessThanOrEqual(sizes.filter { $0.isRegularFile == true }.reduce(0) { $0 + ($1.fileSize ?? 0) }, 4096)
    }
    func testContentionFallsBackImmediatelyWithoutWaitingForAnotherPublisher() throws {
        let root = try temporaryDirectory(); defer { try? FileManager.default.removeItem(at: root) }
        let source = try sourcePackage(root, count: 16)
        let cacheRoot = root.appendingPathComponent("cache", isDirectory: true)
        let fingerprint = identity().fingerprint
        let cache = try NativeGraphCodeCache(root: cacheRoot, identity: fingerprint)
        XCTAssertNotNil(cache.store(package: source, metadata: Data("feeds".utf8), key: "stage"))
        let fd = Darwin.open(cacheRoot.appendingPathComponent(".lock").path, O_RDWR)
        guard fd >= 0 else { return XCTFail("Could not open test publisher lock.") }
        defer { flock(fd, LOCK_UN); Darwin.close(fd) }
        XCTAssertEqual(flock(fd, LOCK_EX | LOCK_NB), 0)
        let started = ProcessInfo.processInfo.systemUptime
        XCTAssertNil(cache.package(for: "stage"))
        XCTAssertNil(cache.store(package: source, metadata: Data("feeds".utf8), key: "stage"))
        XCTAssertThrowsError(try NativeGraphCodeCache(root: cacheRoot, identity: fingerprint))
        XCTAssertLessThan(ProcessInfo.processInfo.systemUptime - started, 1,
            "An optional cache must not wait for another process or delay Abort.")
        flock(fd, LOCK_UN)
        XCTAssertNotNil(cache.package(for: "stage"))
    }
    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("NativeCodeCacheTest-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
    private func sourcePackage(_ root: URL, count: Int) throws -> URL {
        let source = root.appendingPathComponent("source.mpsgraphpackage", isDirectory: true)
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try Data(repeating: 3, count: count).write(to: source.appendingPathComponent("code.bin"))
        return source
    }
}

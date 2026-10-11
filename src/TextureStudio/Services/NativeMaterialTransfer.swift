import CryptoKit
import Foundation

/// Streams model files through URLSession. Only small metadata is held in RAM;
/// large uploads use LFS and downloads publish verified complete directories.
struct NativeMaterialTransfer: Sendable {
    static let revision = "73ab49a0cc0de5ea70e7aa94fb1a7234dd59ab35"
    static let weightsName = "pbrnxt_402236.pth"
    static let weightsBytes: Int64 = 349_493_406
    static let weightsSHA256 = "3f25b03e950c6199b53a3e1581296831e71555e1928ad209232b757f75153b7d"
    typealias Sender = @Sendable (URLRequest, URL?) async throws -> (Data, HTTPURLResponse)
    typealias Downloader = @Sendable (URLRequest, URL, Int64) async throws -> Void
    let token: String?
    let send: Sender
    let fetch: Downloader
    let catalogURL: URL
    init(token: String? = NativeHuggingFaceService.savedToken(), send: @escaping Sender = Self.sendRequest,
         fetch: @escaping Downloader = Self.fetchRequest, catalogURL: URL = Self.defaultCatalogURL) {
        self.token = token; self.send = send; self.fetch = fetch; self.catalogURL = catalogURL
    }

    static func hash(_ file: URL) throws -> String {
        let reader = try FileHandle(forReadingFrom: file)
        defer { try? reader.close() }
        var digest = SHA256()
        while let bytes = try reader.read(upToCount: 1_048_576), !bytes.isEmpty {
            try Task.checkCancellation(); digest.update(data: bytes)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func installBase(at directory: URL) async throws -> String {
        let destination = directory.standardizedFileURL
        guard (try? destination.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else {
            throw StudioError("Choose a real directory for the base model.")
        }
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        let file = destination.appendingPathComponent(weightsName)
        if FileManager.default.fileExists(atPath: file.path) {
            try verifyBase(file)
        } else {
            let url = URL(string: "https://media.githubusercontent.com/media/aaf6aa/PBRnxt/\(revision)/pretrained_models/\(weightsName)")!
            let artifact = ModelDownloadArtifact(relativePath: weightsName, url: url, byteCount: weightsBytes, sha256: weightsSHA256)
            let stage = destination.appendingPathComponent(".base-download-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: stage) }
            try await ModelDownloadService().fetch(artifact, to: stage, progress: { _ in })
            try verifyBase(stage)
            try Task.checkCancellation()
            try FileManager.default.moveItem(at: stage, to: file)
        }
        return try json(["directory": destination.path, "model_directory": destination.path,
            "weights_sha256": weightsSHA256, "weights_bytes": weightsBytes])
    }

    static func verifyBase(_ file: URL) throws {
        let values = try file.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey, .isSymbolicLinkKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true,
              values.fileSize == Int(weightsBytes), try hash(file) == weightsSHA256 else {
            throw StudioError("The pinned PBRnxt base weights failed their size or SHA-256 identity check.")
        }
    }

    static func removeBase(at directory: URL) throws -> String {
        let file = directory.appendingPathComponent(weightsName)
        try verifyBase(file)
        try FileManager.default.removeItem(at: file)
        return try json(["directory": directory.path, "removed": true, "bytes_reclaimed": weightsBytes, "can_redownload": true])
    }

    func download(repository: String, revision: String, to directory: URL) async throws -> String {
        guard HuggingFaceUpload.validRepository(repository), Self.isRevision(revision) else {
            throw StudioError("Choose a model repository and its exact recorded revision.")
        }
        guard !FileManager.default.fileExists(atPath: directory.path) else {
            let ownership = try Self.object(directory.appendingPathComponent(".texture-studio-native-model.json"))
            guard ownership["repository"] as? String == repository, ownership["revision"] as? String == revision else {
                throw StudioError("Choose a new download directory; the existing directory belongs to another model.")
            }
            let (configuration, _) = try NativeMaterialPackage.verify(directory)
            let name = configuration["checkpoint_filename"] as? String ?? "adapter.safetensors"
            return try NativeMaterialCheckpoint.inspect(at: directory.appendingPathComponent(name))
        }
        let parent = directory.deletingLastPathComponent()
        try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
        let stage = parent.appendingPathComponent(".material-download-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: stage) }
        let manifestURL = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/SHA256SUMS.json")!
        try await downloadFile(manifestURL, to: stage.appendingPathComponent("SHA256SUMS.json"), maxBytes: 1_048_576)
        let checksums = try Self.object(stage.appendingPathComponent("SHA256SUMS.json"))
        guard !checksums.isEmpty, checksums.count <= 32 else { throw StudioError("Invalid model package inventory.") }
        for name in checksums.keys.sorted() {
            guard NativeMaterialPackage.allowedFile(name), let expected = checksums[name] as? String,
                  Self.isDigest(expected) else { throw StudioError("The model package contains an unsupported file.") }
            let file = stage.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            let remote = URL(string: "https://huggingface.co/\(repository)/resolve/\(revision)/\(name)")!
            try await downloadFile(remote, to: file, maxBytes: name.hasSuffix(".safetensors") ? 4_294_967_296 : 4_194_304)
            guard try Self.hash(file) == expected else { throw StudioError("Downloaded model failed SHA-256 verification.") }
        }
        let (configuration, hashes) = try NativeMaterialPackage.verify(stage)
        let name = configuration["checkpoint_filename"] as? String ?? "adapter.safetensors"
        let digest = hashes[name] as? String ?? ""
        try Self.writeJSON(["repository": repository, "revision": revision, "sha256": digest], to: stage.appendingPathComponent(".texture-studio-native-model.json"))
        try Task.checkCancellation()
        try FileManager.default.moveItem(at: stage, to: directory)
        try register(["repository": repository, "revision": revision, "sha256": digest,
            "architecture": configuration["architecture"] ?? NSNull(), "model_family": configuration["model_family"] ?? "pbrnxt",
            "target": configuration["target"] ?? NSNull(), "checkpoint_filename": name])
        return try NativeMaterialCheckpoint.inspect(at: directory.appendingPathComponent(name))
    }

    func upload(package: URL, repository: String, isPublic: Bool) async throws -> String {
        guard HuggingFaceUpload.validRepository(repository), token != nil else {
            throw StudioError("Save a Hugging Face token and choose an owner/model repository.")
        }
        let (configuration, hashes) = try NativeMaterialPackage.verify(package)
        let manifestDigest = try Self.hash(package.appendingPathComponent("SHA256SUMS.json"))
        let parts = repository.split(separator: "/").map(String.init)
        _ = try await request("https://huggingface.co/api/repos/create", method: "POST",
            object: ["name": parts[1], "organization": parts[0], "type": "model", "private": !isPublic], accepted: [200, 201, 409])
        let (infoBytes, _) = try await request("https://huggingface.co/api/models/\(repository)")
        guard let info = try JSONSerialization.jsonObject(with: infoBytes) as? [String: Any],
              info["private"] as? Bool == !isPublic else {
            throw StudioError("The existing repository visibility differs from the selected upload visibility.")
        }
        let names = (Array(hashes.keys) + ["SHA256SUMS.json"]).sorted()
        var records: [[String: Any]] = []
        for name in names {
            let file = package.appendingPathComponent(name)
            let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            let reader = try FileHandle(forReadingFrom: file)
            let sample = try reader.read(upToCount: 512) ?? Data(); try reader.close()
            records.append(["path": name, "size": size, "sample": sample.base64EncodedString()])
        }
        let (preupload, _) = try await request("https://huggingface.co/api/models/\(repository)/preupload/main", method: "POST", object: ["files": records])
        guard let pre = try JSONSerialization.jsonObject(with: preupload) as? [String: Any],
              let modes = pre["files"] as? [[String: Any]], modes.count == names.count,
              Set(modes.compactMap { $0["path"] as? String }) == Set(names) else {
            throw StudioError("The Hub upload inventory differs from the verified model package.")
        }
        let summary = configuration["schema"] as? String == NativeCompactMaterialModel.schema
            ? "Upload Texture Studio standalone compact material model" : "Upload Texture Studio native material checkpoint and LoRA"
        var lines: [[String: Any]] = [["key": "header", "value": ["summary": summary]]]
        for name in names {
            try Task.checkCancellation()
            let file = package.appendingPathComponent(name), mode = modes.first { $0["path"] as? String == name }?["uploadMode"] as? String
            let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            let checksum = try Self.hash(file)
            guard checksum == (name == "SHA256SUMS.json" ? manifestDigest : hashes[name] as? String) else {
                throw StudioError("The model package changed during upload; export it again.")
            }
            if mode == "lfs" {
                try await uploadLFS(file, repository: repository, checksum: checksum, size: size)
                lines.append(["key": "lfsFile", "value": ["path": name, "algo": "sha256", "oid": checksum, "size": size]])
            } else if mode == "regular", size <= 8_388_608 {
                lines.append(["key": "file", "value": ["path": name, "encoding": "base64", "content": try Data(contentsOf: file).base64EncodedString()]])
            } else { throw StudioError("The Hub returned an unsupported model transfer mode.") }
        }
        // Recheck files after transfer; another process must not publish changed bytes.
        let (_, currentHashes) = try NativeMaterialPackage.verify(package)
        guard NSDictionary(dictionary: currentHashes).isEqual(to: hashes),
              try Self.hash(package.appendingPathComponent("SHA256SUMS.json")) == manifestDigest else {
            throw StudioError("The model package changed during upload; export it again.")
        }
        let body = try lines.map { try Self.json($0) }.joined(separator: "\n") + "\n"
        let (result, _) = try await request("https://huggingface.co/api/models/\(repository)/commit/main", method: "POST",
            data: Data(body.utf8), contentType: "application/x-ndjson")
        guard let commit = try JSONSerialization.jsonObject(with: result) as? [String: Any],
              let revision = commit["commitOid"] as? String, Self.isRevision(revision) else { throw StudioError("The Hub did not return a completed model commit.") }
        let name = configuration["checkpoint_filename"] as? String ?? "adapter.safetensors"
        let digest = hashes[name] as? String ?? ""
        try register(["repository": repository, "revision": revision, "target": configuration["target"] ?? NSNull(),
            "architecture": configuration["architecture"] ?? NSNull(), "model_family": configuration["model_family"] ?? "pbrnxt",
            "checkpoint_filename": name, "sha256": digest])
        return try Self.json(["repository": repository, "revision": revision, "url": "https://huggingface.co/\(repository)",
            "commit_url": commit["commitUrl"] ?? "https://huggingface.co/\(repository)/commit/\(revision)",
            "sha256": digest, "private": !isPublic, "source_photos_uploaded": false])
    }

    private func uploadLFS(_ file: URL, repository: String, checksum: String, size: Int) async throws {
        let (body, _) = try await request("https://huggingface.co/\(repository).git/info/lfs/objects/batch", method: "POST",
            object: ["operation": "upload", "transfers": ["basic", "multipart"], "hash_algo": "sha256",
                "objects": [["oid": checksum, "size": size]]], contentType: "application/vnd.git-lfs+json")
        guard let batch = try JSONSerialization.jsonObject(with: body) as? [String: Any],
              let objects = batch["objects"] as? [[String: Any]], objects.count == 1,
              let item = objects.first, item["oid"] as? String == checksum,
              item["size"] as? Int == size, item["error"] == nil else {
            throw StudioError("The Hub could not prepare the model object upload.")
        }
        guard let actionValue = item["actions"] else { return } // Object already exists in LFS.
        guard let actions = actionValue as? [String: Any] else { throw StudioError("Invalid model object upload actions.") }
        guard let upload = actions["upload"] as? [String: Any] else {
            guard actions.isEmpty else { throw StudioError("Missing model object upload action.") }
            return
        }
        guard let href = upload["href"] as? String else { throw StudioError("Missing model upload destination.") }
        let headers = upload["header"] as? [String: String] ?? [:]
        if let chunkText = headers["chunk_size"] {
            guard let chunkSize = Int(chunkText), chunkSize > 0 else { throw StudioError("Invalid multipart model chunk size.") }
            let partKeys = headers.keys.filter { Int($0) != nil }.sorted { Int($0)! < Int($1)! }
            guard partKeys.count == size / chunkSize + (size % chunkSize == 0 ? 0 : 1),
                  partKeys.enumerated().allSatisfy({ Int($0.element) == $0.offset + 1 }) else { throw StudioError("Invalid multipart model transfer layout.") }
            let input = try FileHandle(forReadingFrom: file); defer { try? input.close() }
            var parts: [[String: Any]] = []
            for part in partKeys {
                try Task.checkCancellation()
                let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("material-upload-\(UUID().uuidString)")
                defer { try? FileManager.default.removeItem(at: temporary) }
                let handle = try FileHandle(forWritingTo: {
                    FileManager.default.createFile(atPath: temporary.path, contents: nil); return temporary
                }())
                defer { try? handle.close() }
                var remaining = min(chunkSize, size - (Int(part)! - 1) * chunkSize)
                while remaining > 0 {
                    let bytes = try input.read(upToCount: min(1_048_576, remaining)) ?? Data()
                    guard !bytes.isEmpty else { throw StudioError("Model file was truncated during upload.") }
                    try handle.write(contentsOf: bytes); remaining -= bytes.count
                }
                try handle.close()
                let (_, response) = try await signedRequest(headers[part]!, method: "PUT", file: temporary)
                guard let etag = response.value(forHTTPHeaderField: "ETag"), !etag.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw StudioError("Missing multipart upload receipt.") }
                parts.append(["partNumber": Int(part)!, "etag": etag])
            }
            _ = try await signedRequest(href, method: "POST", object: ["oid": checksum, "parts": parts])
        } else {
            _ = try await signedRequest(href, method: "PUT", file: file, headers: headers)
        }
        if let verify = actions["verify"] as? [String: Any], let href = verify["href"] as? String {
            _ = try await signedRequest(href, method: "POST", object: ["oid": checksum, "size": size], headers: verify["header"] as? [String: String] ?? [:])
        }
    }

    private func downloadFile(_ url: URL, to file: URL, maxBytes: Int64) async throws {
        guard Self.validHTTPS(url), maxBytes > 0 else { throw StudioError("Model transfers require HTTPS and a bounded size.") }
        var request = URLRequest(url: url, timeoutInterval: 120)
        if let token { request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization") }
        try Task.checkCancellation()
        try await fetch(request, file, maxBytes)
        let values = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey, .fileSizeKey])
        guard values.isRegularFile == true, values.isSymbolicLink != true,
              Int64(values.fileSize ?? 0) <= maxBytes else {
            throw StudioError("The model download failed or exceeded its declared format budget.")
        }
        try Task.checkCancellation()
    }

    private func request(_ path: String, method: String = "GET", object: [String: Any]? = nil,
                         data: Data? = nil, contentType: String = "application/json", accepted: Set<Int> = [200, 201]) async throws -> (Data, HTTPURLResponse) {
        var headers = ["Content-Type": contentType, "Accept": contentType]
        if let token { headers["Authorization"] = "Bearer " + token }
        return try await signedRequest(path, method: method, object: object, data: data, headers: headers, accepted: accepted)
    }

    private func signedRequest(_ path: String, method: String, file: URL? = nil, object: [String: Any]? = nil,
                               data: Data? = nil, headers: [String: String] = [:], accepted: Set<Int> = [200, 201, 204]) async throws -> (Data, HTTPURLResponse) {
        guard let url = URL(string: path), Self.validHTTPS(url) else { throw StudioError("Model transfers require HTTPS.") }
        var request = URLRequest(url: url, timeoutInterval: 120)
        request.httpMethod = method; request.allHTTPHeaderFields = headers
        if let object {
            request.httpBody = try JSONSerialization.data(withJSONObject: object)
            if request.value(forHTTPHeaderField: "Content-Type") == nil { request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        }
        else { request.httpBody = data }
        try Task.checkCancellation()
        let result = try await send(request, file)
        try Task.checkCancellation()
        guard result.1.url.map(Self.validHTTPS) == true,
              accepted.contains(result.1.statusCode) else { throw StudioError("Model transfer failed (HTTP \(result.1.statusCode)).") }
        return result
    }

    static func sendRequest(_ request: URLRequest, file: URL?) async throws -> (Data, HTTPURLResponse) {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForResource = 3600
        let session = URLSession(configuration: configuration, delegate: MaterialHTTPSDelegate(), delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        let result: (Data, URLResponse)
        if let file { result = try await session.upload(for: request, fromFile: file) }
        else { result = try await session.data(for: request) }
        guard let response = result.1 as? HTTPURLResponse, response.url.map(Self.validHTTPS) == true else { throw StudioError("Invalid model transfer response.") }
        return (result.0, response)
    }

    private func register(_ record: [String: Any]) throws {
        let catalog = catalogURL
        var records = (try? Self.object(catalog)["models"] as? [[String: Any]]) ?? []
        records.removeAll { $0["repository"] as? String == record["repository"] as? String }
        records.append(record)
        try FileManager.default.createDirectory(at: catalog.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Self.writeJSON(["models": records], to: catalog)
    }

    static func isDigest(_ value: String) -> Bool { value.count == 64 && value.allSatisfy { "0123456789abcdef".contains($0) } }
    static let defaultCatalogURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        .appendingPathComponent("Texture Studio/material-hub-catalog.json")
    static func isRevision(_ value: String) -> Bool { [40, 64].contains(value.count) && value.allSatisfy { "0123456789abcdef".contains($0) } }
    static func validHTTPS(_ url: URL) -> Bool { url.scheme == "https" && url.host?.isEmpty == false && url.user == nil && url.password == nil }
    static func fetchRequest(_ request: URLRequest, to file: URL, maxBytes: Int64) async throws {
        let available = try file.deletingLastPathComponent().resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage
        guard let available, available > 67_108_864 else { throw StudioError("Free disk space is insufficient for the model download.") }
        let limit = min(maxBytes, available - 67_108_864)
        let delegate = MaterialDownloadDelegate(maxBytes: limit)
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 120; configuration.timeoutIntervalForResource = 3600
        let session = URLSession(configuration: configuration, delegate: delegate, delegateQueue: nil)
        defer { session.invalidateAndCancel() }
        let temporary: URL, response: URLResponse
        do { (temporary, response) = try await session.download(for: request, delegate: delegate) }
        catch {
            if delegate.exceededSize { throw StudioError("The model download exceeded its format or available disk budget.") }
            throw error
        }
        defer { try? FileManager.default.removeItem(at: temporary) }
        guard let http = response as? HTTPURLResponse, http.statusCode == 200,
              http.url.map(Self.validHTTPS) == true,
              Int64(try temporary.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0) <= limit else {
            throw StudioError("The model download failed or exceeded its declared format budget.")
        }
        try Task.checkCancellation()
        try FileManager.default.moveItem(at: temporary, to: file)
    }
    static func object(_ file: URL) throws -> [String: Any] {
        try object(Data(contentsOf: file))
    }
    static func object(_ data: Data) throws -> [String: Any] {
        guard let result = try JSONSerialization.jsonObject(with: data) as? [String: Any] else { throw StudioError("Invalid material JSON metadata.") }
        return result
    }
    static func json(_ object: [String: Any]) throws -> String { String(decoding: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), as: UTF8.self) }
    static func writeJSON(_ object: [String: Any], to file: URL) throws {
        try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys, .prettyPrinted]).write(to: file, options: .atomic)
    }
}

/// Reject protocol downgrades, and never forward a repository token to a
/// redirected object-storage host. Signed object URLs carry their own access.
class MaterialHTTPSDelegate: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        guard let url = request.url, NativeMaterialTransfer.validHTTPS(url) else { completionHandler(nil); return }
        var next = request
        if url.host != task.originalRequest?.url?.host {
            let ordinary = Set(["accept", "accept-encoding", "content-type", "content-length", "user-agent"])
            for name in next.allHTTPHeaderFields?.keys ?? Dictionary<String, String>().keys where !ordinary.contains(name.lowercased()) {
                next.setValue(nil, forHTTPHeaderField: name)
            }
        }
        completionHandler(next)
    }
}

final class MaterialDownloadDelegate: MaterialHTTPSDelegate, URLSessionDownloadDelegate, @unchecked Sendable {
    let maxBytes: Int64
    private let lock = NSLock()
    private var violation = false
    var exceededSize: Bool { lock.withLock { violation } }
    init(maxBytes: Int64) { self.maxBytes = maxBytes }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask,
                    didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64) {
        if totalBytesWritten > maxBytes || totalBytesExpectedToWrite > maxBytes {
            lock.withLock { violation = true }
            downloadTask.cancel()
        }
    }
    func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {}
}

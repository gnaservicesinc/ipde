import CryptoKit
import Foundation

enum NativeCheckpointError: LocalizedError {
    case invalid(String)
    var errorDescription: String? {
        switch self { case .invalid(let reason): return "Invalid material checkpoint: \(reason)" }
    }
}

/// An immutable safetensors snapshot. Tensor payloads remain their original
/// little-endian bytes; parsing does not cast, normalize, or rewrite weights.
struct NativeSafetensors: Sendable {
    struct Tensor: Decodable, Sendable {
        let dtype: String
        let shape: [Int]
        let dataOffsets: [Int]
        enum CodingKeys: String, CodingKey { case dtype, shape; case dataOffsets = "data_offsets" }
    }

    private struct Header: Decodable {
        struct Key: CodingKey {
            let stringValue: String
            var intValue: Int? { nil }
            init(stringValue: String) { self.stringValue = stringValue }
            init?(intValue: Int) { return nil }
        }
        let metadata: [String: String]
        let tensors: [String: Tensor]
        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy: Key.self)
            var tensors: [String: Tensor] = [:]
            var metadata: [String: String] = [:]
            for key in values.allKeys {
                if key.stringValue == "__metadata__" {
                    metadata = try values.decode([String: String].self, forKey: key)
                } else {
                    tensors[key.stringValue] = try values.decode(Tensor.self, forKey: key)
                }
            }
            self.metadata = metadata
            self.tensors = tensors
        }
    }

    let bytes: Data
    let metadata: [String: String]
    let tensors: [String: Tensor]
    let sha256: String
    let payloadStart: Int

    init(bytes: Data, expectedSHA256: String? = nil) throws {
        try Task.checkCancellation()
        guard bytes.count >= 8 else { throw NativeCheckpointError.invalid("truncated file") }
        let length = bytes.withUnsafeBytes { UInt64(littleEndian: $0.loadUnaligned(as: UInt64.self)) }
        guard length >= 2, length <= 100_000_000, length <= UInt64(bytes.count - 8) else {
            throw NativeCheckpointError.invalid("invalid or truncated header length")
        }
        let payloadStart = 8 + Int(length)
        let headerBytes = bytes.subdata(in: 8..<payloadStart)
        guard headerBytes.first == UInt8(ascii: "{") else { throw NativeCheckpointError.invalid("header must begin with an object") }
        try StrictJSONKeys.validate(headerBytes, integerFields: ["shape", "data_offsets"])
        let header: Header
        do { header = try JSONDecoder().decode(Header.self, from: headerBytes) }
        catch { throw NativeCheckpointError.invalid("header types or tensor descriptors are invalid") }
        let digest = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        try Task.checkCancellation()
        if let expectedSHA256, digest != expectedSHA256 {
            throw NativeCheckpointError.invalid("selected checkpoint changed; reload it before use")
        }
        var cursor = 0
        for (name, tensor) in header.tensors.sorted(by: {
            let lhs = $0.value.dataOffsets.first ?? -1, rhs = $1.value.dataOffsets.first ?? -1
            if lhs != rhs { return lhs < rhs }
            // Empty tensors share a boundary with the next nonempty tensor.
            return ($0.value.dataOffsets.last ?? -1) < ($1.value.dataOffsets.last ?? -1)
        }) {
            try Task.checkCancellation()
            guard let width = Self.byteWidth[tensor.dtype], tensor.dataOffsets.count == 2,
                  tensor.shape.allSatisfy({ $0 >= 0 }), tensor.dataOffsets[0] == cursor,
                  tensor.dataOffsets[1] >= cursor, tensor.dataOffsets[1] <= bytes.count - payloadStart else {
                throw NativeCheckpointError.invalid("tensor \(name) has invalid type, shape, or offsets")
            }
            var elements = 1
            for dimension in tensor.shape {
                let product = elements.multipliedReportingOverflow(by: dimension)
                guard !product.overflow else { throw NativeCheckpointError.invalid("tensor dimensions overflow") }
                elements = product.partialValue
            }
            let size = elements.multipliedReportingOverflow(by: width)
            guard !size.overflow, size.partialValue == tensor.dataOffsets[1] - cursor else {
                throw NativeCheckpointError.invalid("tensor \(name) byte count differs from its shape")
            }
            cursor = tensor.dataOffsets[1]
        }
        guard cursor == bytes.count - payloadStart else { throw NativeCheckpointError.invalid("payload contains unclaimed bytes") }
        self.bytes = bytes
        self.metadata = header.metadata
        self.tensors = header.tensors
        self.sha256 = digest
        self.payloadStart = payloadStart
    }

    init(contentsOf url: URL, expectedSHA256: String? = nil) throws {
        // A copied snapshot binds metadata, hashes and tensor bytes to the same
        // contents even if another process replaces the file after inspection.
        try self.init(bytes: Data(contentsOf: url), expectedSHA256: expectedSHA256)
    }

    func tensorBytes(named name: String) throws -> Data {
        guard let tensor = tensors[name] else { throw NativeCheckpointError.invalid("missing tensor \(name)") }
        return bytes.subdata(in: (payloadStart + tensor.dataOffsets[0])..<(payloadStart + tensor.dataOffsets[1]))
    }

    /// Check finiteness directly from exponent bits, preserving even signed
    /// zero and subnormal encodings rather than passing through a Float cast.
    func validateFiniteFloatingPoint() throws {
        try bytes.withUnsafeBytes { raw in
            for (name, tensor) in tensors {
                try Task.checkCancellation()
                let start = payloadStart + tensor.dataOffsets[0], end = payloadStart + tensor.dataOffsets[1]
                switch tensor.dtype {
                case "F64":
                    for offset in stride(from: start, to: end, by: 8) {
                        if (offset - start) & 0x003f_ffff == 0 { try Task.checkCancellation() }
                        let bits = UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt64.self))
                        if bits & 0x7ff0_0000_0000_0000 == 0x7ff0_0000_0000_0000 { throw NativeCheckpointError.invalid("tensor \(name) contains nonfinite values") }
                    }
                case "F32":
                    for offset in stride(from: start, to: end, by: 4) {
                        if (offset - start) & 0x003f_ffff == 0 { try Task.checkCancellation() }
                        let bits = UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
                        if bits & 0x7f80_0000 == 0x7f80_0000 { throw NativeCheckpointError.invalid("tensor \(name) contains nonfinite values") }
                    }
                case "F16", "BF16":
                    let exponent: UInt16 = tensor.dtype == "F16" ? 0x7c00 : 0x7f80
                    for offset in stride(from: start, to: end, by: 2) {
                        if (offset - start) & 0x003f_ffff == 0 { try Task.checkCancellation() }
                        let bits = UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: offset, as: UInt16.self))
                        if bits & exponent == exponent { throw NativeCheckpointError.invalid("tensor \(name) contains nonfinite values") }
                    }
                default: break
                }
            }
        }
    }

    private static let byteWidth = ["BOOL": 1, "U8": 1, "I8": 1, "U16": 2, "I16": 2,
        "U32": 4, "I32": 4, "U64": 8, "I64": 8, "F16": 2, "BF16": 2, "F32": 4, "F64": 8]

    static func encoded(tensors: [String: NativeTensor], metadata: [String: String]) throws -> Data {
        try Task.checkCancellation()
        var header: [String: Any] = ["__metadata__": metadata], payload = Data()
        for name in tensors.keys.sorted() {
            guard name != "__metadata__", let tensor = tensors[name] else { throw NativeCheckpointError.invalid("reserved tensor name") }
            let start = payload.count
            payload.append(tensor.bytes)
            header[name] = ["dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [start, payload.count]]
        }
        var json = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        json.append(Data(repeating: 32, count: (8 - json.count % 8) % 8))
        var length = UInt64(json.count).littleEndian
        var bytes = withUnsafeBytes(of: &length) { Data($0) }
        bytes.append(json); bytes.append(payload)
        let checked = try Self(bytes: bytes)
        try checked.validateFiniteFloatingPoint()
        return bytes
    }

    static func write(tensors: [String: NativeTensor], metadata: [String: String], to url: URL) throws {
        let bytes = try encoded(tensors: tensors, metadata: metadata)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try bytes.write(to: url, options: .atomic)
        guard try Data(contentsOf: url) == bytes else { throw NativeCheckpointError.invalid("written checkpoint failed byte verification") }
    }

    func nativeTensors() throws -> [String: NativeTensor] {
        try Dictionary(uniqueKeysWithValues: tensors.map { key, value in
            (key, NativeTensor(dtype: value.dtype, shape: value.shape, bytes: try tensorBytes(named: key)))
        })
    }
}

/// Checkpoint metadata inspection used by the app without launching Python,
/// importing PyTorch, or loading the full model into GPU memory.
enum NativeMaterialCheckpoint {
    private struct Configuration: Decodable {
        struct Base: Decodable { let sha256: String }
        struct Layer: Decodable { let weightShape: [Int]; let rank: Int; let alpha: Double
            enum CodingKeys: String, CodingKey { case weightShape = "weight_shape"; case rank, alpha }
        }
        let schema: String, architecture: String, target: String
        let step: Int, trainingSize: Int
        let imagePadding: Bool, imageResizing: Bool
        let base: Base
        let scope: String?
        let modelName: String?
        let layers: [String: Layer]?
        enum CodingKeys: String, CodingKey {
            case schema, architecture, target, step, base, scope, layers
            case trainingSize = "training_size", imagePadding = "image_padding", imageResizing = "image_resizing"
            case modelName = "model_name"
        }
    }

    static func inspect(at requestedURL: URL, expectedSHA256: String? = nil) throws -> String {
        try Task.checkCancellation()
        var url = requestedURL.standardizedFileURL.resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue {
            let full = url.appendingPathComponent("model.safetensors")
            url = FileManager.default.fileExists(atPath: full.path) ? full : url.appendingPathComponent("adapter.safetensors")
        }
        guard url.pathExtension == "safetensors" else { throw NativeCheckpointError.invalid("choose a material .safetensors file or export directory") }
        let snapshot = try NativeSafetensors(contentsOf: url, expectedSHA256: expectedSHA256)
        try snapshot.validateFiniteFloatingPoint()
        guard let configurationJSON = snapshot.metadata["configuration"], let data = configurationJSON.data(using: .utf8) else {
            throw NativeCheckpointError.invalid("recorded material configuration is missing")
        }
        try StrictJSONKeys.validate(data, integerFields: ["step", "training_size", "rank", "weight_shape",
                                                       "network_width", "input_channels", "output_channels", "initialization_seed"])
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NativeCheckpointError.invalid("material configuration must be an object")
        }
        if object["schema"] as? String == NativeCompactMaterialModel.schema {
            try NativeCompactMaterialModel.validateCheckpoint(snapshot, configuration: object)
            guard let architecture = object["architecture"] as? String, let family = object["model_family"] as? String,
                  let target = object["target"] as? String, let step = object["step"] as? Int,
                  let size = object["training_size"] as? Int, let inputChannels = object["input_channels"] as? Int,
                  let outputChannels = object["output_channels"] as? Int, let width = object["network_width"] as? Int,
                  let seed = object["initialization_seed"] as? UInt64, let initialDigest = object["initial_weights_sha256"] as? String else {
                throw NativeCheckpointError.invalid("compact checkpoint identity or training metadata is missing")
            }
            let information: [String: Any] = ["checkpoint_path": url.path, "sha256": snapshot.sha256,
                "schema": NativeCompactMaterialModel.schema, "architecture": architecture,
                "model_family": family, "target": target, "step": step,
                "compatible": true, "variant": "full", "supports_training_warm_start": true,
                "supports_studio_inference": true, "refinement_policy": "native_compact_full_training",
                "training_size": size, "input_channels": inputChannels,
                "output_channels": outputChannels, "network_width": width,
                "from_scratch": true, "base_required": false, "base": NSNull(),
                "initialization_seed": seed, "initial_weights_sha256": initialDigest,
                "model_name": object["model_name"] ?? NSNull(), "scope": "full-model", "validation": object["validation"] ?? NSNull()]
            try Task.checkCancellation()
            return String(decoding: try JSONSerialization.data(withJSONObject: information, options: [.sortedKeys]), as: UTF8.self)
        }
        let configuration: Configuration
        do { configuration = try JSONDecoder().decode(Configuration.self, from: data) }
        catch { throw NativeCheckpointError.invalid("material configuration types are invalid") }
        let full = configuration.schema == "texture-studio-material-checkpoint-v1"
        guard full || configuration.schema == "texture-studio-material-lora-v1",
              configuration.architecture == "pbrnxt-native-v1",
              ["height", "normal", "roughness"].contains(configuration.target),
              configuration.step >= 0, configuration.base.sha256.count == 64,
              configuration.base.sha256.allSatisfy({ "0123456789abcdef".contains($0) }),
              !configuration.imagePadding, !configuration.imageResizing,
              (256...8192).contains(configuration.trainingSize), configuration.trainingSize % 64 == 0,
              !snapshot.tensors.isEmpty else {
            throw NativeCheckpointError.invalid("expected a complete recorded native material checkpoint")
        }
        if !full { try validateAdapter(configuration, snapshot: snapshot) }
        let information: [String: Any] = ["checkpoint_path": url.path, "sha256": snapshot.sha256,
            "schema": configuration.schema, "architecture": configuration.architecture,
            "target": configuration.target, "step": configuration.step, "compatible": true,
            "variant": full ? "full" : "lora", "supports_training_warm_start": true,
            "supports_studio_inference": true, "refinement_policy": "native_material_lora",
            "base": object["base"]!, "training_size": configuration.trainingSize,
            "model_name": configuration.modelName as Any? ?? NSNull(),
            "scope": object["scope"] ?? NSNull(), "validation": object["validation"] ?? NSNull()]
        try Task.checkCancellation()
        return String(decoding: try JSONSerialization.data(withJSONObject: information, options: [.sortedKeys]), as: UTF8.self)
    }

    private static func validateAdapter(_ configuration: Configuration, snapshot: NativeSafetensors) throws {
        guard let layers = configuration.layers, !layers.isEmpty,
              let scope = configuration.scope, ["final-map", "map-decoder"].contains(scope) else {
            throw NativeCheckpointError.invalid("adapter layer specifications or scope are missing")
        }
        let expected = Set(layers.keys.flatMap { [$0 + ".lora_A", $0 + ".lora_B"] })
        guard Set(snapshot.tensors.keys) == expected else { throw NativeCheckpointError.invalid("adapter tensor identities differ from its configuration") }
        let branch = ["normal": 1, "roughness": 2, "height": 3][configuration.target]!
        var prefixes = ["ups.\(branch)."]
        if scope == "map-decoder" { prefixes += ["gen.m_dec_\(branch).", "gen.m_tail_\(branch)."] }
        for (name, spec) in layers {
            guard [2, 4].contains(spec.weightShape.count), spec.weightShape.allSatisfy({ $0 > 0 }),
                  spec.rank > 0, spec.alpha.isFinite, spec.alpha > 0,
                  prefixes.contains(where: { name == String($0.dropLast()) || name.hasPrefix($0) }) else {
                throw NativeCheckpointError.invalid("adapter \(name) has invalid dimensions or target")
            }
            var input = 1
            for dimension in spec.weightShape.dropFirst() {
                let product = input.multipliedReportingOverflow(by: dimension)
                guard !product.overflow else { throw NativeCheckpointError.invalid("adapter dimensions overflow") }
                input = product.partialValue
            }
            guard snapshot.tensors[name + ".lora_A"]?.shape == [spec.rank, input],
                  snapshot.tensors[name + ".lora_B"]?.shape == [spec.weightShape[0], spec.rank],
                  snapshot.tensors[name + ".lora_A"]?.dtype == "F32",
                  snapshot.tensors[name + ".lora_B"]?.dtype == "F32" else {
                throw NativeCheckpointError.invalid("adapter \(name) tensor shapes or precision differ")
            }
        }
    }
}

/// Foundation accepts duplicate JSON keys; reject them before decoding a
/// checkpoint so there is a single unambiguous tensor/configuration identity.
private enum StrictJSONKeys {
    static func validate(_ data: Data, integerFields: Set<String> = []) throws {
        var scanner = Scanner(bytes: Array(data), integerFields: integerFields)
        try scanner.value(depth: 0)
        scanner.whitespace()
        guard scanner.position == scanner.bytes.count else { throw NativeCheckpointError.invalid("trailing JSON data") }
    }
    private struct Scanner {
        let bytes: [UInt8]
        let integerFields: Set<String>
        var position = 0
        mutating func whitespace() { while position < bytes.count && [9, 10, 13, 32].contains(bytes[position]) { position += 1 } }
        mutating func take(_ byte: UInt8) -> Bool {
            whitespace()
            if position < bytes.count && bytes[position] == byte { position += 1; return true }
            return false
        }
        mutating func string() throws -> String {
            whitespace()
            let start = position
            guard take(34) else { throw NativeCheckpointError.invalid("invalid JSON string") }
            while position < bytes.count {
                let byte = bytes[position]; position += 1
                if byte == 92 {
                    guard position < bytes.count else { throw NativeCheckpointError.invalid("truncated JSON escape") }
                    position += 1
                } else if byte == 34 {
                    do { return try JSONDecoder().decode(String.self, from: Data(bytes[start..<position])) }
                    catch { throw NativeCheckpointError.invalid("invalid JSON string encoding") }
                }
            }
            throw NativeCheckpointError.invalid("unterminated JSON string")
        }
        mutating func value(depth: Int, requiresInteger: Bool = false) throws {
            guard depth < 128 else { throw NativeCheckpointError.invalid("JSON nesting is too deep") }
            whitespace()
            guard position < bytes.count else { throw NativeCheckpointError.invalid("truncated JSON") }
            if take(123) {
                var keys = Set<String>()
                if take(125) { return }
                repeat {
                    let key = try string()
                    guard keys.insert(key).inserted, take(58) else { throw NativeCheckpointError.invalid("duplicate or malformed JSON object key") }
                    try value(depth: depth + 1, requiresInteger: integerFields.contains(key))
                    if take(125) { return }
                    guard take(44) else { throw NativeCheckpointError.invalid("invalid JSON object separator") }
                } while true
            } else if take(91) {
                if take(93) { return }
                repeat {
                    try value(depth: depth + 1, requiresInteger: requiresInteger)
                    if take(93) { return }
                    guard take(44) else { throw NativeCheckpointError.invalid("invalid JSON array separator") }
                } while true
            } else if bytes[position] == 34 {
                guard !requiresInteger else { throw NativeCheckpointError.invalid("expected an integer JSON value") }
                _ = try string()
            }
            else {
                let start = position
                while position < bytes.count && ![9, 10, 13, 32, 44, 93, 125].contains(bytes[position]) { position += 1 }
                guard start < position else { throw NativeCheckpointError.invalid("invalid JSON value") }
                if requiresInteger {
                    let token = bytes[start..<position]
                    let digits = token.first == 45 ? token.dropFirst() : token[...]
                    guard !digits.isEmpty, digits.allSatisfy({ (48...57).contains($0) }) else {
                        throw NativeCheckpointError.invalid("expected an integer JSON value")
                    }
                }
                // JSONDecoder validates numbers, booleans and null when it
                // decodes the complete document after this identity pass.
            }
        }
    }
}

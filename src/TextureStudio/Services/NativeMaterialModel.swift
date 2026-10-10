import Accelerate
import CryptoKit
import Foundation
import Metal
import MetalPerformanceShaders
import MetalPerformanceShadersGraph

struct NativeMaterialPrediction: Sendable {
    let width: Int, height: Int, channels: Int
    /// Planar NCHW, the direct Float32 network output. No clipping/stretching.
    let values: [Float]
}

/// The pinned PBRnxt SCUNetV2 + RRDB mapping, expressed directly as an Apple
/// MPSGraph. The graph retains every generator decoder and the chosen complete
/// output branch; only the previously declared final 4x enlargement is omitted.
final class NativeMaterialModel: @unchecked Sendable {
    static let pinnedSHA256 = "3f25b03e950c6199b53a3e1581296831e71555e1928ad209232b757f75153b7d"
    static let pinnedFilename = "pbrnxt_402236.pth"
    struct AdapterLayer: Codable, Sendable {
        let weightShape: [Int], rank: Int
        let alpha: Float
        enum CodingKeys: String, CodingKey { case weightShape = "weight_shape"; case rank, alpha }
    }
    struct Architecture: Sendable {
        let dim: Int, heads: Int, encoderBlocks: Int, decoderBlocks: Int, fusionBlocks: Int, rrdbBlocks: Int, rrdbWidth: Int, growth: Int
        static let pinned = Architecture(dim: 96, heads: 1, encoderBlocks: 2, decoderBlocks: 2, fusionBlocks: 4, rrdbBlocks: 12, rrdbWidth: 32, growth: 32)
        /// Used by numeric tests of the complete operation sequence.
        static let test = Architecture(dim: 4, heads: 1, encoderBlocks: 2, decoderBlocks: 2, fusionBlocks: 2, rrdbBlocks: 1, rrdbWidth: 2, growth: 2)
    }
    let baseWeights: [String: NativeTensor]
    private(set) var adapterWeights: [String: NativeTensor]
    let layers: [String: AdapterLayer]
    var configuration: [String: Any]
    let baseSHA256: String
    let architecture: Architecture
    private var cached: Program?

    init(baseWeights: [String: NativeTensor], adapterWeights: [String: NativeTensor] = [:],
         layers: [String: AdapterLayer] = [:], configuration: [String: Any] = [:], baseSHA256: String,
         architecture: Architecture = .pinned) throws {
        guard !baseWeights.isEmpty else { throw StudioError("The material model has no learned tensors.") }
        for (name, tensor) in baseWeights {
            var byteCount = tensor.dtype == "I64" ? 8 : 4
            for dimension in tensor.shape {
                let product = byteCount.multipliedReportingOverflow(by: dimension)
                guard dimension > 0, !product.overflow else { throw StudioError("Material tensor dimensions overflow: \(name)") }
                byteCount = product.partialValue
            }
            guard ["F32", "I64", "I32"].contains(tensor.dtype), !tensor.shape.isEmpty,
                  tensor.bytes.count == byteCount else {
                throw StudioError("Material tensor precision or dimensions are invalid: \(name)")
            }
            if name.hasSuffix(".relative_position_index") {
                guard tensor.dtype == "I64", tensor.shape == [64, 64] else { throw StudioError("The material attention position index has invalid storage.") }
                try tensor.bytes.withUnsafeBytes { bytes in
                    for offset in stride(from: 0, to: bytes.count, by: 8) {
                        let index = bytes.loadUnaligned(fromByteOffset: offset, as: Int64.self)
                        guard (0..<225).contains(index) else { throw StudioError("The material attention position index exceeds its learned table.") }
                    }
                }
            }
        }
        let expectedFactors = Set(layers.keys.flatMap { [$0 + ".lora_A", $0 + ".lora_B"] })
        guard Set(adapterWeights.keys) == expectedFactors else { throw StudioError("Material adapter factors differ from their recorded layers.") }
        for (name, layer) in layers {
            guard let weight = baseWeights[name + ".weight"], weight.dtype == "F32",
                  [2, 4].contains(weight.shape.count), weight.shape == layer.weightShape,
                  !name.contains(".dwconv."), layer.rank > 0,
                  layer.alpha.isFinite, layer.alpha > 0 else {
                throw StudioError("Material adapter layer differs from its exact base weight: \(name)")
            }
            let incoming = weight.shape.dropFirst().reduce(1, *)
            let aCount = layer.rank.multipliedReportingOverflow(by: incoming)
            let bCount = layer.rank.multipliedReportingOverflow(by: weight.shape[0])
            let aBytes = aCount.partialValue.multipliedReportingOverflow(by: 4)
            let bBytes = bCount.partialValue.multipliedReportingOverflow(by: 4)
            guard !aCount.overflow, !bCount.overflow, !aBytes.overflow, !bBytes.overflow,
                  let a = adapterWeights[name + ".lora_A"], let b = adapterWeights[name + ".lora_B"],
                  a.dtype == "F32", b.dtype == "F32", a.shape == [layer.rank, incoming],
                  b.shape == [weight.shape[0], layer.rank], a.bytes.count == aBytes.partialValue,
                  b.bytes.count == bBytes.partialValue else {
                throw StudioError("Material adapter factors have invalid native dimensions: \(name)")
            }
        }
        self.baseWeights = baseWeights; self.adapterWeights = adapterWeights
        self.layers = layers; self.configuration = configuration; self.baseSHA256 = baseSHA256
        self.architecture = architecture
    }

    static func load(checkpointURL: URL?, expectedSHA256: String? = nil, baseURL: URL,
                     target: String, scope: String = "final-map", rank: Int = 8, alpha: Float = 8,
                     training: Bool = false, seed: UInt64 = 17) throws -> NativeMaterialModel {
        var configuration: [String: Any] = [:], adapters: [String: NativeTensor] = [:]
        var specs: [String: AdapterLayer] = [:], base: [String: NativeTensor], digest: String
        if let checkpointURL {
            _ = try NativeMaterialCheckpoint.inspect(at: checkpointURL, expectedSHA256: expectedSHA256)
            let checkpoint = try NativeSafetensors(contentsOf: resolvedCheckpoint(checkpointURL), expectedSHA256: expectedSHA256)
            configuration = try JSONSerialization.jsonObject(with: Data(checkpoint.metadata["configuration"]!.utf8)) as! [String: Any]
            guard configuration["target"] as? String == target,
                  !training || configuration["schema"] as? String != "texture-studio-material-lora-v1" || configuration["scope"] as? String == scope else {
                throw StudioError("The selected checkpoint target or trained layer scope differs from this operation.")
            }
            if configuration["schema"] as? String == "texture-studio-material-checkpoint-v1" {
                base = try checkpoint.nativeTensors(); digest = checkpoint.sha256
                configuration["base"] = ["sha256": digest, "architecture": "pbrnxt-native-v1", "name": "Texture Studio custom material base"]
            } else {
                adapters = try checkpoint.nativeTensors()
                let specData = try JSONSerialization.data(withJSONObject: configuration["layers"]!)
                specs = try JSONDecoder().decode([String: AdapterLayer].self, from: specData)
                let loaded = try readWeights(baseURL); base = loaded.0; digest = loaded.1
                guard (configuration["base"] as? [String: Any])?["sha256"] as? String == digest else {
                    throw StudioError("The material adapter requires its exact recorded base weights.")
                }
            }
        } else {
            let loaded = try readWeights(baseURL); base = loaded.0; digest = loaded.1
            configuration = ["base": ["sha256": digest, "architecture": "pbrnxt-native-v1", "name": "PBRnxt material mapping"], "step": 0]
        }
        if training && specs.isEmpty {
            guard rank > 0, alpha.isFinite, alpha > 0, ["final-map", "map-decoder"].contains(scope),
                  let branch = ["normal": 1, "roughness": 2, "height": 3][target] else { throw StudioError("Invalid material adapter parameters.") }
            let prefixes = ["ups.\(branch)."] + (scope == "map-decoder" ? ["gen.m_dec_\(branch).", "gen.m_tail_\(branch)."] : [])
            var random = NativeMaterialRandom(seed: seed)
            for key in base.keys.sorted() where key.hasSuffix(".weight") && prefixes.contains(where: { key.hasPrefix($0) }) && !key.contains(".dwconv.") {
                let weight = base[key]!
                guard [2, 4].contains(weight.shape.count), weight.dtype == "F32" else { continue }
                let layer = String(key.dropLast(7))
                var incoming = 1
                for dimension in weight.shape.dropFirst() {
                    let count = incoming.multipliedReportingOverflow(by: dimension)
                    guard dimension > 0, !count.overflow else { throw StudioError("Material adapter dimensions overflow: \(layer)") }
                    incoming = count.partialValue
                }
                let aCount = rank.multipliedReportingOverflow(by: incoming)
                let bCount = rank.multipliedReportingOverflow(by: weight.shape[0])
                guard !aCount.overflow, !bCount.overflow,
                      aCount.partialValue <= Int.max / 4, bCount.partialValue <= Int.max / 4 else {
                    throw StudioError("Material adapter allocation dimensions overflow: \(layer)")
                }
                let bound = 1 / sqrt(Float(incoming))
                let a = (0..<aCount.partialValue).map { _ in (random.unit() * 2 - 1) * bound }
                specs[layer] = AdapterLayer(weightShape: weight.shape, rank: rank, alpha: alpha)
                adapters[layer + ".lora_A"] = .floats(a, shape: [rank, incoming])
                adapters[layer + ".lora_B"] = .floats([Float](repeating: 0, count: bCount.partialValue), shape: [weight.shape[0], rank])
            }
            guard !specs.isEmpty else { throw StudioError("The recorded model has no trained layers for this target.") }
        }
        configuration["architecture"] = "pbrnxt-native-v1"
        configuration["target"] = target; configuration["scope"] = scope
        configuration["schema"] = specs.isEmpty ? "texture-studio-material-checkpoint-v1" : "texture-studio-material-lora-v1"
        return try NativeMaterialModel(baseWeights: base, adapterWeights: adapters, layers: specs, configuration: configuration, baseSHA256: digest)
    }

    static func resolvedCheckpoint(_ url: URL) -> URL {
        var isDirectory: ObjCBool = false
        if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue {
            let full = url.appendingPathComponent("model.safetensors")
            return FileManager.default.fileExists(atPath: full.path) ? full : url.appendingPathComponent("adapter.safetensors")
        }
        return url
    }
    private static func readWeights(_ url: URL) throws -> ([String: NativeTensor], String) {
        if url.pathExtension == "safetensors" {
            let snapshot = try NativeSafetensors(contentsOf: url)
            try snapshot.validateFiniteFloatingPoint()
            return (try snapshot.nativeTensors(), snapshot.sha256)
        }
        let loaded = try NativeTorchCheckpoint.load(url: url, expectedSHA256: pinnedSHA256)
        return (loaded.tensors, loaded.sha256)
    }
    func predict(rgb: [Float], width: Int, height: Int, target: String) throws -> NativeMaterialPrediction {
        try validateInput(rgb, width: width, height: height)
        let program = try program(width: width, height: height, target: target)
        let result = try program.execute(rgb: rgb, adapters: adapterWeights)
        return NativeMaterialPrediction(width: width, height: height, channels: target == "normal" ? 3 : 1, values: result.output)
    }
    func program(width: Int, height: Int, target: String) throws -> Program {
        if let cached, cached.width == width, cached.height == height, cached.target == target { return cached }
        let next = try Program(model: self, width: width, height: height, target: target)
        cached = next; return next
    }
    func updateAdapters(_ adapters: [String: NativeTensor]) { self.adapterWeights = adapters }
    func validateInput(_ rgb: [Float], width: Int, height: Int) throws {
        guard width >= 64, height >= 64, width % 64 == 0, height % 64 == 0,
              rgb.count == width * height * 3, rgb.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
            throw StudioError("Material inference needs finite native RGB on dimensions divisible by 64, without resizing or padding.")
        }
        try Task.checkCancellation()
    }
    func checkpointConfiguration(size: Int, step: Int, validation: [String: Any]? = nil) throws -> [String: Any] {
        var record = configuration
        record["step"] = step; record["training_size"] = size
        record["image_padding"] = false; record["image_resizing"] = false
        record["input_transfer"] = "sRGB diffuse codes / code maximum"
        record["target_transfer"] = "linear numeric source codes / code maximum"
        record["native_runtime"] = "Apple MPSGraph Float32, reduced precision fast math disabled"
        record["layers"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(layers))
        record["trained_utc"] = ISO8601DateFormatter().string(from: Date())
        if let validation { record["validation"] = validation }
        return record
    }
    func fusedWeights() throws -> [String: NativeTensor] {
        var fused = baseWeights
        for (name, specification) in layers {
            try Task.checkCancellation()
            guard let base = baseWeights[name + ".weight"], let a = adapterWeights[name + ".lora_A"],
                  let b = adapterWeights[name + ".lora_B"] else { throw StudioError("Incomplete material adapter state.") }
            let graph = MPSGraph()
            let wa = graph.constant(a.bytes, shape: a.shape.ns, dataType: .float32)
            let wb = graph.constant(b.bytes, shape: b.shape.ns, dataType: .float32)
            let delta = graph.multiplication(graph.matrixMultiplication(primary: wb, secondary: wa, name: nil),
                graph.constant(Double(specification.alpha / Float(specification.rank)), dataType: .float32), name: nil)
            let weight = graph.constant(base.bytes, shape: base.shape.ns, dataType: .float32)
            let sum = graph.addition(weight, graph.reshape(delta, shape: base.shape.ns, name: nil), name: nil)
            let values = try NativeGraphExecution.run(graph, feeds: [:], targets: [sum])[0]
            fused[name + ".weight"] = .floats(values, shape: base.shape)
        }
        return fused
    }

    /// Reuses staged symbolic metadata and compact immutable feeds without
    /// compiling that graph. Disposable compiler owners and loaded executables
    /// release their scratch storage after GPU completion. Unstaged graphs stay fresh.
    final class Program {
        let width: Int, height: Int, target: String
        let frozenStageCount: Int
        private(set) var optimizerPrepared = false
        private let baseWeights: [String: NativeTensor]
        private let layers: [String: AdapterLayer]
        private let architecture: Architecture
        private let adapterShapes: [String: [Int]]
        private let stagesEnabled: Bool
        private let checkpointByteLimit: UInt64?
        private let coarseWorkspaceByteLimit: UInt64
        private let coalescingActiveBlocks: Bool
        private let packages: NativeGraphPackageCache
        private var stagedProgram: GraphProgram?
        typealias Execution = GraphProgram.Execution
        var executionStatistics: NativeGraphExecution.Statistics {
            var result = packages.statistics
            result.finalCommandBufferGPUSeconds = packages.stream.finalCommandBufferGPUSeconds
            result.waitSeconds = packages.stream.waitSeconds
            result.commandBuffers = packages.stream.commandBuffers
            result.maximumInFlightStages = packages.stream.maximumInFlightStages
            return result
        }

        /// Leave retained activations and the admitted model working set outside
        /// the local workspace allowance of a coalesced backward stage.
        static func coarseWorkspaceBudget(capacity: UInt64, checkpointBytes: UInt64, workingBytes: UInt64) -> UInt64 {
            let frozenBytes = min(UInt64(2 * 1_073_741_824), capacity / 16)
            return [min(checkpointBytes, capacity / 2), frozenBytes, workingBytes].reduce(capacity) {
                $0 > $1 ? $0 - $1 : 0
            }
        }

        init(model: NativeMaterialModel, width: Int, height: Int, target: String, staged: Bool = true, checkpointByteLimit: UInt64? = nil,
             coalesceActiveBlocks: Bool? = nil) throws {
            guard ["height", "roughness", "normal"].contains(target), width >= 64, height >= 64,
                  width % 64 == 0, height % 64 == 0 else { throw StudioError("Unsupported native material grid or target.") }
            self.width = width; self.height = height; self.target = target
            baseWeights = model.baseWeights; layers = model.layers; architecture = model.architecture
            adapterShapes = model.adapterWeights.mapValues(\.shape)
            stagesEnabled = staged
            let environment = ProcessInfo.processInfo.environment
            // Enable adaptive active-block admission through 1K by default.
            // Explicit overrides take priority; environment value 0 opts out.
            let coalescing = staged && max(width, height) <= 1024 &&
                (coalesceActiveBlocks ?? (environment["TEXTURE_STUDIO_COALESCE_ACTIVE_BLOCKS"] != "0"))
            coalescingActiveBlocks = coalescing
            let capacity = MachineResources.current.maximumTrainingBytes
            // Spend only half the estimated spare capacity on checkpoints,
            // leaving the rest for transient stage and compiler allocations.
            // Frozen features retain their separate 2 GiB storage limit.
            let workingBytes = NativeMaterialTrainer.estimatedWorkingBytes(model: model, size: max(width, height))
            let spareBytes = capacity > workingBytes ? capacity - workingBytes : 0
            // Full 2K stages require substantially larger backend workspace
            // than their logical tensors. Preserve that headroom instead of
            // allowing checkpoints to turn GPU work into system swapping.
            // Coarse active blocks retain fewer boundary activations but need
            // larger local VJP workspace. Bound their checkpoint pool to leave
            // headroom for the adaptive block admission policy below.
            let checkpointCap = UInt64((coalescing ? 8 : max(width, height) > 1024 ? 6 : 24) * 1_073_741_824)
            let defaultCheckpointBytes = min(checkpointCap, min(capacity / 2, spareBytes / 2))
            let checkpointBudget = checkpointByteLimit ?? defaultCheckpointBytes
            self.checkpointByteLimit = checkpointBudget
            let coarseWorkspaceBudget = Self.coarseWorkspaceBudget(capacity: capacity,
                checkpointBytes: checkpointBudget, workingBytes: workingBytes)
            coarseWorkspaceByteLimit = coarseWorkspaceBudget
            let codeCache: NativeGraphCodeCache?
            let osBuild = NativeGraphCodeIdentity.osBuild
            if staged, environment["TEXTURE_STUDIO_DISABLE_PROGRAM_CACHE"] != "1",
               osBuild != "unknown",
               let executableSHA256 = NativeGraphCodeIdentity.executableSHA256,
               let device = MTLCreateSystemDefaultDevice() {
                let a = model.architecture
                let identity = NativeGraphCodeIdentity(executableSHA256: executableSHA256,
                    osVersion: ProcessInfo.processInfo.operatingSystemVersionString, osBuild: osBuild,
                    metalName: device.name, metalRegistryID: device.registryID, maximumTrainingBytes: capacity,
                    baseSHA256: model.baseSHA256, width: width, height: height, target: target,
                    architecture: [a.dim, a.heads, a.encoderBlocks, a.decoderBlocks, a.fusionBlocks, a.rrdbBlocks, a.rrdbWidth, a.growth,
                        coalescing ? 1 : 0] + (coalescing ? [Int(coarseWorkspaceBudget)] : []),
                    tensors: model.baseWeights.map { NativeGraphCodeIdentity.Tensor(name: $0.key, dtype: $0.value.dtype, shape: $0.value.shape) },
                    layers: model.layers.map { NativeGraphCodeIdentity.Layer(name: $0.key, weightShape: $0.value.weightShape,
                        rank: $0.value.rank, alphaBits: $0.value.alpha.bitPattern) })
                let directory = environment["TEXTURE_STUDIO_PROGRAM_CACHE_DIRECTORY"].map { URL(fileURLWithPath: $0, isDirectory: true) }
                    ?? NativeGraphCodeCache.defaultRoot
                codeCache = directory.flatMap { try? NativeGraphCodeCache(root: $0, identity: identity.fingerprint) }
            } else { codeCache = nil }
            let packageCache = try NativeGraphPackageCache(persistent: codeCache)
            packages = packageCache
            let engine = try autoreleasepool {
                try GraphProgram(baseWeights: model.baseWeights, layers: model.layers, architecture: model.architecture,
                    adapters: model.adapterWeights, width: width, height: height, target: target, staged: staged,
                    packageCache: staged ? packageCache : nil, checkpointByteLimit: checkpointBudget,
                    coarseWorkspaceByteLimit: coarseWorkspaceBudget,
                    coalescingActiveBlocks: coalescing)
            }
            frozenStageCount = engine.frozenStageCount
            if staged { stagedProgram = engine }
        }
        func execute(rgb: [Float], adapters: [String: NativeTensor], reference: [Float]? = nil,
                     learningRate: Float? = nil, step: Int = 1, optimizerState: [String: NativeTensor] = [:],
                     gradientsOnly: Bool = false,
                     featureKey: String? = nil, checkCancellation: () throws -> Void = { try Task.checkCancellation() },
                     onStage: (Int, Int) -> Void = { _, _ in },
                     onOperation: (String, Int, Int) -> Void = { _, _, _ in }) throws -> Execution {
            let activity = ProcessInfo.processInfo.beginActivity(options: .userInitiated, reason: "Computing the requested material model")
            defer { ProcessInfo.processInfo.endActivity(activity) }
            do {
                try checkCancellation()
                return try autoreleasepool {
                    let engine: GraphProgram
                    if let stagedProgram {
                        engine = stagedProgram
                        onOperation("Reusing model execution graph", 1, 1)
                    } else {
                        // A discarded engine must be rebuilt with the same
                        // structure as its already cached stage packages.
                        try GraphProgram.validateAdapters(adapters, expectedShapes: adapterShapes)
                        onOperation("Building model execution graph", 0, 1)
                        engine = try GraphProgram(baseWeights: baseWeights, layers: layers, architecture: architecture,
                            adapters: adapters, width: width, height: height, target: target, staged: stagesEnabled,
                            packageCache: stagesEnabled ? packages : nil, checkpointByteLimit: checkpointByteLimit,
                            coarseWorkspaceByteLimit: coarseWorkspaceByteLimit,
                            coalescingActiveBlocks: coalescingActiveBlocks)
                        if stagesEnabled { stagedProgram = engine }
                        onOperation("Building model execution graph", 1, 1)
                    }
                    defer {
                        // Errors and Abort must join submitted GPU work before
                        // releasing its graph, inputs or checkpoint storage.
                        try? packages.stream.finish()
                        engine.releaseCompilerOwner()
                        optimizerPrepared = optimizerPrepared || engine.optimizerPrepared
                    }
                    try engine.validateAdapters(adapters)
                    let result = try engine.execute(rgb: rgb, adapters: adapters, reference: reference, learningRate: learningRate,
                        step: step, optimizerState: optimizerState, gradientsOnly: gradientsOnly, featureKey: featureKey,
                        checkCancellation: checkCancellation, onStage: onStage, onOperation: onOperation)
                    try packages.stream.finish()
                    return result
                }
            } catch {
                // Preparation can stop after creating only some derivative
                // plans. Never reuse incomplete metadata or cached features.
                stagedProgram = nil
                throw error
            }
        }
    }

    final class GraphProgram {
        private let packageCache: NativeGraphPackageCache?
        private var packageCompiler: GraphProgram?
        private var compilerJobs = 0
        private var compilerRetainedBytes: UInt64 = 0
        private var checkpointNames: [String]?
        private var generatorAnchor: MPSGraphTensor?
        private let checkpointByteLimit: UInt64?
        private let coarseWorkspaceByteLimit: UInt64
        let graph = MPSGraph()
        let width: Int, height: Int, target: String
        let input: MPSGraphTensor
        private(set) var output: MPSGraphTensor!
        private(set) var parameterFeeds: [String: MPSGraphTensor] = [:]
        private(set) var parameters: [String: NativeTensor] = [:]
        private let baseWeights: [String: NativeTensor]
        private let layers: [String: AdapterLayer]
        private let architecture: Architecture
        private var constants: [String: MPSGraphTensor] = [:]
        private var auxiliaryFeeds: [MPSGraphTensor: NativeTensor] = [:]
        private var immutableData: [MPSGraphTensor: MPSGraphTensorData]?
        private var maskLabels: [String: MPSGraphTensor] = [:]
        private(set) var targetInput: MPSGraphTensor?
        private(set) var loss: MPSGraphTensor?
        private(set) var valueLoss: MPSGraphTensor?
        private(set) var gradientLoss: MPSGraphTensor?
        private var gradients: [String: MPSGraphTensor] = [:]
        private var optimizerFeeds: [String: MPSGraphTensor] = [:]
        private var optimizerOutputs: [String: MPSGraphTensor] = [:]
        private var learningRate: MPSGraphTensor?
        private var optimizerStep: MPSGraphTensor?
        private var executableCache: [String: MPSGraphExecutable] = [:]
        private var frozenStages: [[(source: MPSGraphTensor, feed: MPSGraphTensor)]] = []
        private let stagesEnabled: Bool
        private let coalescingActiveBlocks: Bool
        private var isolatedWorkspaceStages = Set<Int>()
        private var dependencies: [MPSGraphTensor: Set<MPSGraphTensor>] = [:]
        private var featureCache: [String: [MPSGraphTensor: MPSGraphTensorData]] = [:]
        private var featureOrder: [String] = []
        private var featureBytes: UInt64 = 0
        private var adapterInfluencedFeeds = Set<MPSGraphTensor>()
        private var coalescingBlock = false
        private var finalLayerOnly: Bool {
            !layers.isEmpty && layers.keys.allSatisfy { $0 == "ups.\(["normal": 1, "roughness": 2, "height": 3][target]!).model.10" }
        }
        var frozenStageCount: Int { frozenStages.count }
        var optimizerPrepared: Bool { learningRate != nil }
        fileprivate func releaseCompilerOwner() {
            packageCompiler = nil
            compilerJobs = 0
            compilerRetainedBytes = 0
        }
        fileprivate func validateAdapters(_ adapters: [String: NativeTensor]) throws {
            try Self.validateAdapters(adapters, expectedShapes: parameterFeeds.mapValues { $0.shape!.map(\.intValue) })
            // Compiler owners use these tensors only for placeholder shapes;
            // retaining the current factors also releases the preceding values.
            parameters = adapters
        }
        fileprivate static func validateAdapters(_ adapters: [String: NativeTensor], expectedShapes: [String: [Int]]) throws {
            guard Set(adapters.keys) == Set(expectedShapes.keys) else {
                throw StudioError("Native material adapter factors changed for this execution graph.")
            }
            for (name, expectedShape) in expectedShapes {
                guard let tensor = adapters[name], tensor.dtype == "F32",
                      tensor.shape == expectedShape else {
                    throw StudioError("Native material adapter dimensions or precision changed: \(name).")
                }
                let byteCount = try tensor.shape.reduce(4) { bytes, dimension in
                    let product = bytes.multipliedReportingOverflow(by: dimension)
                    guard dimension > 0, !product.overflow else {
                        throw StudioError("Native material adapter storage dimensions overflow: \(name).")
                    }
                    return product.partialValue
                }
                guard tensor.bytes.count == byteCount else {
                    throw StudioError("Native material adapter storage changed: \(name).")
                }
            }
        }
        convenience init(model: NativeMaterialModel, width: Int, height: Int, target: String, staged: Bool = true, checkpointByteLimit: UInt64? = nil,
                         coarseWorkspaceByteLimit: UInt64? = nil,
                         coalescingActiveBlocks: Bool = false) throws {
            guard ["height", "roughness", "normal"].contains(target), width >= 64, height >= 64,
                  width % 64 == 0, height % 64 == 0 else { throw StudioError("Unsupported native material grid or target.") }
            let workspaceBudget = coarseWorkspaceByteLimit ?? Program.coarseWorkspaceBudget(
                capacity: MachineResources.current.maximumTrainingBytes,
                checkpointBytes: checkpointByteLimit ?? UInt64(2 * 1_073_741_824),
                workingBytes: NativeMaterialTrainer.estimatedWorkingBytes(model: model, size: max(width, height)))
            try self.init(baseWeights: model.baseWeights, layers: model.layers, architecture: model.architecture,
                adapters: model.adapterWeights, width: width, height: height, target: target, staged: staged,
                checkpointByteLimit: checkpointByteLimit, coarseWorkspaceByteLimit: workspaceBudget,
                coalescingActiveBlocks: coalescingActiveBlocks)
        }
        fileprivate init(baseWeights: [String: NativeTensor], layers: [String: AdapterLayer], architecture: Architecture,
                     adapters: [String: NativeTensor], width: Int, height: Int, target: String, staged: Bool = true,
                     packageCache: NativeGraphPackageCache? = nil, checkpointByteLimit: UInt64? = nil, checkpointNames: [String]? = nil,
                     coarseWorkspaceByteLimit: UInt64,
                     coalescingActiveBlocks: Bool = false) throws {
            guard ["height", "roughness", "normal"].contains(target), width >= 64, height >= 64,
                  width % 64 == 0, height % 64 == 0 else { throw StudioError("Unsupported native material grid or target.") }
            // Share immutable base storage without retaining the model that
            // owns this program, avoiding a cycle through the program cache.
            self.baseWeights = baseWeights; self.layers = layers; self.architecture = architecture
            self.stagesEnabled = staged; self.packageCache = packageCache; self.checkpointByteLimit = checkpointByteLimit; self.checkpointNames = checkpointNames
            self.coarseWorkspaceByteLimit = coarseWorkspaceByteLimit
            self.coalescingActiveBlocks = coalescingActiveBlocks && max(width, height) <= 1024
            self.width = width; self.height = height; self.target = target
            input = graph.placeholder(shape: [1, 3, height, width].ns, dataType: .float32, name: "native_rgb")
            for name in adapters.keys.sorted() {
                let tensor = adapters[name]!
                parameters[name] = tensor
                parameterFeeds[name] = graph.placeholder(shape: tensor.shape.ns, dataType: .float32, name: name)
            }
            output = try build(input)
            guard output.shape?.map(\.intValue) == [1, target == "normal" ? 3 : 1, height, width] else {
                throw StudioError("The material graph changed native output dimensions.")
            }
        }
        private func shape(_ tensor: MPSGraphTensor) -> [Int] { tensor.shape!.map(\.intValue) }
        private func c(_ value: Double) -> MPSGraphTensor { graph.constant(value, dataType: .float32) }
        private func add(_ lhs: MPSGraphTensor, _ rhs: MPSGraphTensor) -> MPSGraphTensor { graph.addition(lhs, rhs, name: nil) }
        private func mul(_ lhs: MPSGraphTensor, _ rhs: MPSGraphTensor) -> MPSGraphTensor { graph.multiplication(lhs, rhs, name: nil) }
        private func sub(_ lhs: MPSGraphTensor, _ rhs: MPSGraphTensor) -> MPSGraphTensor { graph.subtraction(lhs, rhs, name: nil) }
        private func div(_ lhs: MPSGraphTensor, _ rhs: MPSGraphTensor) -> MPSGraphTensor { graph.division(lhs, rhs, name: nil) }
        private func reshape(_ tensor: MPSGraphTensor, _ shape: [Int]) -> MPSGraphTensor { graph.reshape(tensor, shape: shape.ns, name: nil) }
        private func permute(_ tensor: MPSGraphTensor, _ axes: [Int]) -> MPSGraphTensor { graph.transpose(tensor, permutation: axes.ns, name: nil) }
        private func slice(_ tensor: MPSGraphTensor, _ axis: Int, _ start: Int, _ length: Int) -> MPSGraphTensor { graph.sliceTensor(tensor, dimension: axis, start: start, length: length, name: nil) }
        private func cat(_ tensors: [MPSGraphTensor], _ axis: Int) -> MPSGraphTensor { graph.concatTensors(tensors, dimension: axis, name: nil) }
        private func tensor(_ name: String, shape required: [Int]? = nil) throws -> MPSGraphTensor {
            if let value = constants[name] { return value }
            guard let weight = baseWeights[name], required == nil || weight.shape == required else { throw StudioError("Material model tensor missing or shape differs: \(name)") }
            let dtype: MPSDataType = weight.dtype == "I64" ? .int64 : weight.dtype == "I32" ? .int32 : .float32
            // Immutable weights are feeds, avoiding repeated serialization and
            // constant folding of the entire base during every stage compile.
            var value = graph.placeholder(shape: weight.shape.ns, dataType: dtype, name: name)
            auxiliaryFeeds[value] = weight
            if name.hasSuffix(".weight"), let spec = layers[String(name.dropLast(7))] {
                let layer = String(name.dropLast(7))
                guard let a = parameterFeeds[layer + ".lora_A"], let b = parameterFeeds[layer + ".lora_B"] else { throw StudioError("Missing material LoRA factors.") }
                value = add(value, reshape(mul(graph.matrixMultiplication(primary: b, secondary: a, name: nil), c(Double(spec.alpha / Float(spec.rank)))), weight.shape))
            }
            constants[name] = value; return value
        }
        private func conv(_ value: MPSGraphTensor, _ name: String, stride: Int = 1, padding: Int? = nil, groups: Int = 1) throws -> MPSGraphTensor {
            guard let weight = baseWeights[name + ".weight"], weight.dtype == "F32", weight.shape.count == 4,
                  shape(value)[1] == weight.shape[1] * groups else { throw StudioError("Invalid learned material convolution: \(name)") }
            let descriptor = MPSGraphConvolution2DOpDescriptor(strideInX: stride, strideInY: stride, dilationRateInX: 1, dilationRateInY: 1,
                groups: groups, paddingLeft: padding ?? weight.shape[3] / 2, paddingRight: padding ?? weight.shape[3] / 2,
                paddingTop: padding ?? weight.shape[2] / 2, paddingBottom: padding ?? weight.shape[2] / 2,
                paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW)!
            var result = graph.convolution2D(value, weights: try tensor(name + ".weight"), descriptor: descriptor, name: name)
            if let bias = baseWeights[name + ".bias"] {
                guard bias.shape == [weight.shape[0]], bias.dtype == "F32" else { throw StudioError("Invalid material convolution bias.") }
                result = add(result, reshape(try tensor(name + ".bias"), [1, weight.shape[0], 1, 1]))
            }
            return result
        }
        private func linear(_ value: MPSGraphTensor, _ name: String, explicitBias: MPSGraphTensor? = nil) throws -> MPSGraphTensor {
            guard let weight = baseWeights[name + ".weight"], weight.dtype == "F32", weight.shape.count == 2,
                  shape(value).last == weight.shape[1] else { throw StudioError("Invalid learned material linear layer: \(name)") }
            var result = graph.matrixMultiplication(primary: value, secondary: permute(try tensor(name + ".weight"), [1, 0]), name: name)
            if let explicitBias { result = add(result, explicitBias) }
            else if baseWeights[name + ".bias"] != nil { result = add(result, try tensor(name + ".bias", shape: [weight.shape[0]])) }
            return result
        }
        private func linearGeluBoundary(_ value: MPSGraphTensor, _ name: String) throws -> MPSGraphTensor {
            if stagesEnabled, !coalescingBlock, layers[name] != nil {
                let preactivation = boundary(try linear(value, name))
                let index = frozenStages.count
                let result = boundary(gelu(preactivation))
                geluStageInputs[index] = preactivation
                return result
            }
            let preactivation = try linear(value, name)
            let index = frozenStages.count
            let result = boundary(gelu(preactivation))
            if stagesEnabled, !coalescingBlock, layers[name] == nil {
                immutableLinearGeluStages[index] = ImmutableLinearGelu(input: value, preactivation: preactivation,
                    weights: try tensor(name + ".weight"))
            }
            return result
        }
        private func norm(_ value: MPSGraphTensor, _ name: String, epsilon: Double) throws -> MPSGraphTensor {
            let axis = shape(value).count - 1, channels = shape(value).last!
            let mean = graph.mean(of: value, axes: [NSNumber(value: axis)], name: nil)
            let centered = sub(value, mean)
            let variance = graph.mean(of: mul(centered, centered), axes: [NSNumber(value: axis)], name: nil)
            return add(mul(div(centered, graph.squareRoot(with: add(variance, c(epsilon)), name: nil)), try tensor(name + ".weight", shape: [channels])), try tensor(name + ".bias", shape: [channels]))
        }
        private func gelu(_ value: MPSGraphTensor) -> MPSGraphTensor {
            mul(mul(value, c(0.5)), add(c(1), graph.erf(with: mul(value, c(1 / sqrt(2))), name: nil)))
        }
        private func leaky(_ value: MPSGraphTensor) -> MPSGraphTensor { graph.leakyReLU(with: value, alpha: 0.2, name: nil) }
        private func convNeXt(_ value: MPSGraphTensor, _ prefix: String) throws -> MPSGraphTensor {
            let channels = shape(value)[1]
            var x = boundary(permute(try conv(value, prefix + ".dwconv", groups: channels), [0, 2, 3, 1]))
            x = boundary(try norm(x, prefix + ".norm", epsilon: 1e-6))
            x = try linearGeluBoundary(x, prefix + ".pwconv1")
            let energy = graph.squareRoot(with: graph.reductionSum(with: mul(x, x), axes: [1, 2], name: nil), name: nil)
            let normalized = boundary(div(energy, add(graph.mean(of: energy, axes: [3], name: nil), c(1e-6))))
            let scaled = boundary(mul(x, normalized))
            let weighted = boundary(mul(try tensor(prefix + ".grn.gamma", shape: [1, 1, 1, channels * 4]), scaled))
            let grnInput = x
            let grnSource = add(add(weighted, try tensor(prefix + ".grn.beta", shape: [1, 1, 1, channels * 4])), grnInput)
            let grnStage = frozenStages.count
            x = boundary(grnSource)
            if stagesEnabled, !coalescingBlock { sameShapeResidualOperands[grnStage] = [weighted, grnInput] }
            x = boundary(try linear(x, prefix + ".pwconv2"))
            x = mul(x, try tensor(prefix + ".gamma", shape: [channels]))
            return add(value, permute(x, [0, 3, 1, 2]))
        }
        private func roll(_ value: MPSGraphTensor, axis: Int, shift: Int) -> MPSGraphTensor {
            let length = shape(value)[axis], offset = ((shift % length) + length) % length
            if offset == 0 { return value }
            return cat([slice(value, axis, length - offset, offset), slice(value, axis, 0, length - offset)], axis)
        }
        private func windows(_ value: MPSGraphTensor) -> MPSGraphTensor {
            let s = shape(value)
            return reshape(permute(reshape(value, [s[0], s[1] / 8, 8, s[2] / 8, 8, s[3]]), [0, 1, 3, 2, 4, 5]), [-1, 64, s[3]])
        }
        private func reverseWindows(_ value: MPSGraphTensor, height: Int, width: Int) -> MPSGraphTensor {
            let channels = shape(value).last!
            return reshape(permute(reshape(value, [1, height / 8, width / 8, 8, 8, channels]), [0, 1, 3, 2, 4, 5]), [1, height, width, channels])
        }
        private func swin(_ value: MPSGraphTensor, _ prefix: String, shifted: Bool) throws -> MPSGraphTensor {
            let s = shape(value), h = s[1], w = s[2], channels = s[3], heads = architecture.heads
            var x = shifted ? roll(roll(value, axis: 1, shift: -4), axis: 2, shift: -4) : value
            x = windows(x)
            let bias = cat([try tensor(prefix + ".msa.q_bias", shape: [channels]), graph.constant(0, shape: [channels].ns, dataType: .float32), try tensor(prefix + ".msa.v_bias", shape: [channels])], 0)
            let qkv = boundary(permute(reshape(try linear(x, prefix + ".msa.embedding_layer", explicitBias: bias), [-1, 64, 3, heads, channels / heads]), [2, 0, 3, 1, 4]))
            let q = reshape(slice(qkv, 0, 0, 1), [-1, heads, 64, channels / heads])
            let k = reshape(slice(qkv, 0, 1, 1), [-1, heads, 64, channels / heads])
            let v = reshape(slice(qkv, 0, 2, 1), [-1, heads, 64, channels / heads])
            func unit(_ tensor: MPSGraphTensor) -> MPSGraphTensor {
                let magnitude = graph.squareRoot(with: graph.reductionSum(with: mul(tensor, tensor), axes: [3], name: nil), name: nil)
                return div(tensor, graph.maximum(magnitude, c(1e-12), name: nil))
            }
            var attention = graph.matrixMultiplication(primary: unit(q), secondary: permute(unit(k), [0, 1, 3, 2]), name: nil)
            let scale = graph.exponent(with: graph.minimum(try tensor(prefix + ".msa.logit_scale", shape: [heads, 1, 1]), c(4.605170185988092), name: nil), name: nil)
            attention = mul(attention, scale)
            let coordinates = try tensor(prefix + ".msa.relative_coords_table", shape: [1, 15, 15, 2])
            let table = try linear(graph.reLU(with: try linear(coordinates, prefix + ".msa.cpb_mlp.0"), name: nil), prefix + ".msa.cpb_mlp.2")
            let indices = reshape(try tensor(prefix + ".msa.relative_position_index", shape: [64, 64]), [4096])
            let biasTable = graph.gather(withUpdatesTensor: reshape(table, [225, heads]), indicesTensor: indices, axis: 0, batchDimensions: 0, name: nil)
            let relative = mul(c(16), graph.sigmoid(with: permute(reshape(biasTable, [64, 64, heads]), [2, 0, 1]), name: nil))
            attention = add(attention, relative)
            if shifted {
                // A constant here makes MPSGraph fold a windowCount×64×64
                // attention mask for EVERY unused block before graph pruning.
                // Feed the exact compact labels so masks exist only at runtime
                // for the current stage, with identical attention semantics.
                let maskKey = "\(h)x\(w)"
                let labelsTensor: MPSGraphTensor
                if let known = maskLabels[maskKey] { labelsTensor = known }
                else {
                    var labels = [Float](repeating: 0, count: h * w)
                    for y in 0..<h { for x in 0..<w {
                        let ry = y < h - 8 ? 0 : y < h - 4 ? 1 : 2
                        let rx = x < w - 8 ? 0 : x < w - 4 ? 1 : 2
                        labels[y * w + x] = Float(ry * 3 + rx)
                    } }
                    labelsTensor = graph.placeholder(shape: [1, h, w, 1].ns, dataType: .float32, name: "shifted_mask_labels_" + maskKey)
                    auxiliaryFeeds[labelsTensor] = .floats(labels, shape: [1, h, w, 1])
                    maskLabels[maskKey] = labelsTensor
                }
                let maskTokens = reshape(windows(labelsTensor), [-1, 64])
                let difference = sub(reshape(maskTokens, [-1, 64, 1]), reshape(maskTokens, [-1, 1, 64]))
                let mask = graph.select(predicate: graph.notEqual(difference, c(0), name: nil), trueTensor: c(-100), falseTensor: c(0), name: nil)
                attention = add(attention, reshape(mask, [-1, 1, 64, 64]))
            }
            attention = boundary(graph.softMax(with: attention, axis: -1, name: nil))
            x = reshape(permute(graph.matrixMultiplication(primary: attention, secondary: v, name: nil), [0, 2, 1, 3]), [-1, 64, channels])
            x = try linear(x, prefix + ".msa.linear")
            x = reverseWindows(x, height: h, width: w)
            if shifted { x = roll(roll(x, axis: 1, shift: 4), axis: 2, shift: 4) }
            x = boundary(add(value, try norm(x, prefix + ".ln1", epsilon: 1e-5)))
            let hidden = try linearGeluBoundary(x, prefix + ".mlp.0")
            let mlp = boundary(try linear(hidden, prefix + ".mlp.2"))
            return add(x, try norm(mlp, prefix + ".ln2", epsilon: 1e-5))
        }
        private func block(_ value: MPSGraphTensor, _ prefix: String, shifted: Bool) throws -> MPSGraphTensor {
            try Task.checkCancellation()
            // A frozen block needs only its terminal features. Cutting its
            // normalization, GELU, GRN and attention separately would compile,
            // reload and synchronize about sixteen packages per block without
            // saving any backward work. The adaptive policy also uses a complete
            // block VJP on adapter paths at up to 1K; larger grids and blocks
            // whose conservative live set is too large retain fine cuts.
            let parameterDependencies = Set(parameterFeeds.values).union(adapterInfluencedFeeds)
            let hasActiveInput = !requiredFeeds([value]).isDisjoint(with: parameterDependencies)
            let hasActiveWeights = layers.keys.contains { $0.hasPrefix(prefix + ".") }
            let blockBytes = UInt64(shape(value).reduce(4, *))
            let frozenForward = !hasActiveInput && !hasActiveWeights
            let workspaceBudget = frozenForward ? MachineResources.current.maximumTrainingBytes : coarseWorkspaceByteLimit
            let canCoalesce = stagesEnabled && (frozenForward || coalescingActiveBlocks) &&
                blockBytes <= workspaceBudget / 24 / 3
            if canCoalesce {
                coalescingBlock = true
                let result: MPSGraphTensor
                do { result = try blockOperations(value, prefix, shifted: shifted) }
                catch { coalescingBlock = false; throw error }
                coalescingBlock = false
                let index = frozenStages.count
                let output = boundary(result)
                // An active block's complete VJP reserves one local workspace.
                // Its small logical boundaries must not permit two such hidden
                // arenas to overlap in the submission window.
                if !frozenForward { isolatedWorkspaceStages.insert(index) }
                return output
            }
            return try blockOperations(value, prefix, shifted: shifted)
        }
        private func blockOperations(_ value: MPSGraphTensor, _ prefix: String, shifted: Bool) throws -> MPSGraphTensor {
            let mixed = boundary(try conv(value, prefix + ".conv1_1", padding: 0)), half = shape(value)[1] / 2
            let convolution = boundary(try convNeXt(slice(mixed, 1, 0, half), prefix + ".conv_block"))
            // Preserve upstream transpose(1,3), including the exchanged H/W.
            let transformer = boundary(try swin(permute(slice(mixed, 1, half, half), [0, 3, 2, 1]), prefix + ".trans_block", shifted: shifted))
            return boundary(add(value, try conv(cat([convolution, permute(transformer, [0, 3, 2, 1])], 1), prefix + ".conv1_2", padding: 0)))
        }
        private func up2(_ value: MPSGraphTensor) -> MPSGraphTensor {
            NativeGraphExecution.nearestNeighbor2(value, graph: graph)
        }
        private func decode(_ body: MPSGraphTensor, _ skips: [MPSGraphTensor], _ branch: Int) throws -> MPSGraphTensor {
            var x = body
            for level in (1...3).reversed() {
                let prefix = "gen.m_dec_\(branch).m_up\(level)"
                x = add(x, skips[level - 1])
                x = leaky(try conv(up2(x), prefix + ".0.up.1"))
                x = boundary(leaky(try conv(x, prefix + ".0.up.3")))
                for i in 0..<architecture.decoderBlocks {
                    x = try block(x, prefix + ".\(i + 1)", shifted: i % 2 == 1)
                }
            }
            return x
        }
        private func rdb(_ value: MPSGraphTensor, _ prefix: String) throws -> MPSGraphTensor {
            let x1 = leaky(try conv(value, prefix + ".conv1.0"))
            let x2 = add(leaky(try conv(cat([value, x1], 1), prefix + ".conv2.0")), try conv(value, prefix + ".conv1x1", padding: 0))
            let x3 = leaky(try conv(cat([value, x1, x2], 1), prefix + ".conv3.0"))
            let x4 = add(leaky(try conv(cat([value, x1, x2, x3], 1), prefix + ".conv4.0")), x2)
            let x5 = try conv(cat([value, x1, x2, x3, x4], 1), prefix + ".conv5.0")
            return boundary(add(value, mul(x5, c(0.2))))
        }
        private func rrdb(_ value: MPSGraphTensor, _ prefix: String) throws -> MPSGraphTensor {
            try Task.checkCancellation()
            var x = value
            for r in 1...3 { x = try rdb(x, prefix + ".RDB\(r)") }
            return boundary(add(value, mul(x, c(0.2))))
        }
        private func build(_ value: MPSGraphTensor) throws -> MPSGraphTensor {
            let specification = architecture
            let branch = ["normal": 1, "roughness": 2, "height": 3][target]!
            let initial = boundary(try conv(value, "gen.m_head"))
            var x = initial, skips: [MPSGraphTensor] = []
            for level in 1...3 {
                let prefix = "gen.m_enc.m_down\(level)"
                for i in 0..<specification.encoderBlocks { x = try block(x, prefix + ".\(i)", shifted: i % 2 == 1) }
                x = boundary(try conv(x, prefix + ".\(specification.encoderBlocks)", stride: 2, padding: 0))
                skips.append(x)
            }
            for i in 0..<specification.encoderBlocks { x = try block(x, "gen.m_body.\(i)", shifted: i % 2 == 1) }
            let decoded = try (0..<4).map { try decode(x, skips, $0) }
            let concatenated = cat(decoded, 1)
            x = boundary(try conv(concatenated, "gen.m_fuse.0"))
            for i in 0..<specification.fusionBlocks { x = try block(x, "gen.m_fuse.\(i + 1)", shifted: i % 2 == 1) }
            let fusionInput = x
            let fusionName = "gen.m_fuse.\(specification.fusionBlocks + 1)"
            let fusionSource = add(try conv(fusionInput, fusionName), concatenated)
            let fusionStage = frozenStages.count
            x = boundary(fusionSource)
            if stagesEnabled, layers[fusionName] == nil, let weight = baseWeights[fusionName + ".weight"] {
                let descriptor = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1,
                    dilationRateInX: 1, dilationRateInY: 1, groups: 1,
                    paddingLeft: weight.shape[3] / 2, paddingRight: weight.shape[3] / 2,
                    paddingTop: weight.shape[2] / 2, paddingBottom: weight.shape[2] / 2,
                    paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW)!
                frozenFusionResiduals[fusionStage] = FrozenFusionResidual(input: fusionInput,
                    residualOperands: decoded, weights: try tensor(fusionName + ".weight"), descriptor: descriptor)
            }
            let generator = try (0..<4).map { branch in
                let branchStage = frozenStages.count
                let branchInput = boundary(add(slice(x, 1, branch * specification.dim, specification.dim), initial))
                if stagesEnabled {
                    tailSliceResiduals[branchStage] = TailSliceResidual(input: x, residual: initial,
                        start: branch * specification.dim, length: specification.dim)
                }
                let name = "gen.m_tail_\(branch).0", convolutionStage = frozenStages.count
                let result = boundary(try conv(branchInput, name))
                if stagesEnabled, layers[name] == nil, let weight = baseWeights[name + ".weight"] {
                    let descriptor = MPSGraphConvolution2DOpDescriptor(strideInX: 1, strideInY: 1,
                        dilationRateInX: 1, dilationRateInY: 1, groups: 1,
                        paddingLeft: weight.shape[3] / 2, paddingRight: weight.shape[3] / 2,
                        paddingTop: weight.shape[2] / 2, paddingBottom: weight.shape[2] / 2,
                        paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW)!
                    frozenFusionResiduals[convolutionStage] = FrozenFusionResidual(input: branchInput,
                        residualOperands: [], weights: try tensor(name + ".weight"), descriptor: descriptor)
                }
                return result
            }
            let generatorStage = frozenStages.count
            let generatorFeatures = boundary(cat(generator, 1))
            if stagesEnabled { tailConcatOperands[generatorStage] = generator }
            generatorAnchor = generatorFeatures
            x = cat([value, generatorFeatures], 1)
            let prefix = "ups.\(branch).model"
            x = boundary(try conv(x, prefix + ".0"))
            let residual = x
            for i in 0..<specification.rrdbBlocks {
                x = try rrdb(x, prefix + ".1.sub.\(i)")
            }
            x = boundary(add(residual, try conv(x, prefix + ".1.sub.\(specification.rrdbBlocks)")))
            // .2 and .5 are the two original nearest2 upsamplers, omitted.
            for index in [3, 6, 8] { x = boundary(leaky(try conv(x, prefix + ".\(index)"))) }
            return try conv(x, prefix + ".10")
        }
        // Boundaries retain the complete native grid and exact forward map.
        // Reverse-stage VJPs carry gradients across all adapted boundaries.
        private func boundary(_ value: MPSGraphTensor) -> MPSGraphTensor {
            if !stagesEnabled || coalescingBlock { return value }
            return freeze([value], name: "native_stage_\(frozenStages.count)")[0]
        }
        private func requiredFeeds(_ targets: [MPSGraphTensor]) -> Set<MPSGraphTensor> {
            func visit(_ tensor: MPSGraphTensor) -> Set<MPSGraphTensor> {
                if let known = dependencies[tensor] { return known }
                let inputs = tensor.operation.inputTensors
                let result = inputs.isEmpty ? Set([tensor]) : inputs.reduce(into: Set<MPSGraphTensor>()) { $0.formUnion(visit($1)) }
                dependencies[tensor] = result
                return result
            }
            return targets.reduce(into: Set<MPSGraphTensor>()) { $0.formUnion(visit($1)) }
        }
        private func freeze(_ sources: [MPSGraphTensor], name: String) -> [MPSGraphTensor] {
            let parameterDependencies = Set(parameterFeeds.values).union(adapterInfluencedFeeds)
            let outputs = sources.enumerated().map { index, source in
                (source: source, feed: graph.placeholder(shape: source.shape, dataType: .float32, name: "\(name)_\(index)"))
            }
            for item in outputs where !requiredFeeds([item.source]).isDisjoint(with: parameterDependencies) {
                adapterInfluencedFeeds.insert(item.feed)
            }
            frozenStages.append(outputs)
            return outputs.map(\.feed)
        }
        private func executeData(key: String, feeds: [MPSGraphTensor: MPSGraphTensorData], targets: [MPSGraphTensor],
                                 checkCancellation: () throws -> Void) throws -> [MPSGraphTensorData] {
            if let packageCache {
                let isolated: Bool
                if key.hasPrefix("forward-"), let index = Int(key.dropFirst("forward-".count)) {
                    isolated = isolatedWorkspaceStages.contains(index)
                } else if key.hasPrefix("reverse-"), let index = Int(key.dropFirst("reverse-".count)) {
                    isolated = isolatedWorkspaceStages.contains(index)
                } else { isolated = false }
                if isolated {
                    // Logical input/output sizes understate a coalesced block's
                    // hidden backend workspace. Never overlap its execution
                    // with another stage, even when replay reuses cached code.
                    try packageCache.stream.finish()
                    try checkCancellation()
                }
                do {
                    let result = try NativeGraphExecution.runPackaged(graph, feeds: feeds, targets: targets, key: key, packages: packageCache, compile: {
                        try self.compilePackage(key: key, feeds: feeds, packages: packageCache)
                    }, checkCancellation: checkCancellation)
                    if isolated { try packageCache.stream.finish(); try checkCancellation() }
                    return result
                } catch {
                    if isolated { try? packageCache.stream.finish() }
                    throw error
                }
            }
            return try NativeGraphExecution.runData(graph, feeds: feeds, targets: targets, cache: &executableCache,
                                                    checkCancellation: checkCancellation)
        }
        /// Keep compiler arenas in a disposable owner, bounded to eight jobs
        /// or 512 MiB of new allocation plus one job. The execution graph only
        /// owns metadata, and this owner is released before any new batch.
        private func compilePackage(key: String, feeds: [MPSGraphTensor: MPSGraphTensorData], packages: NativeGraphPackageCache) throws -> NativeGraphPackageCache.Entry {
            try autoreleasepool {
                if packageCompiler == nil {
                    compilerRetainedBytes = 0
                    packageCompiler = try GraphProgram(baseWeights: baseWeights, layers: layers, architecture: architecture,
                        adapters: parameters, width: width, height: height, target: target, staged: stagesEnabled,
                        coarseWorkspaceByteLimit: coarseWorkspaceByteLimit,
                        coalescingActiveBlocks: coalescingActiveBlocks)
                    compilerJobs = 0
                }
                let owner = packageCompiler!
                let targets = try owner.compilationTargets(key: key)
                let needed = owner.requiredFeeds(targets)
                var current: [String: MPSGraphTensorData] = [:]
                for (tensor, data) in feeds {
                    guard current.updateValue(data, forKey: tensor.operation.name) == nil else {
                        throw StudioError("Native compilation feeds lack unique identities.")
                    }
                }
                var rebound: [MPSGraphTensor: MPSGraphTensorData] = [:]
                for tensor in needed where !tensor.operation.name.isEmpty {
                    if let data = current[tensor.operation.name] {
                        guard tensor.shape?.map(\.intValue) == data.shape.map(\.intValue), tensor.dataType == data.dataType else {
                            throw StudioError("Native disposable compiler feed changed storage: \(tensor.operation.name).")
                        }
                        rebound[tensor] = data
                    } else if owner.graph.placeholderTensors.contains(tensor) {
                        throw StudioError("Native disposable compiler lost a required feed: \(tensor.operation.name).")
                    }
                }
                let before = UInt64(MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0)
                let entry = try NativeGraphExecution.compilePackage(owner.graph, feeds: rebound, targets: targets, key: key, packages: packages)
                let after = UInt64(MTLCreateSystemDefaultDevice()?.currentAllocatedSize ?? 0)
                compilerRetainedBytes += after > before ? after - before : 0
                compilerJobs += 1
                if compilerJobs >= 8 || compilerRetainedBytes >= 512 * 1_048_576 {
                    packageCompiler = nil
                    compilerJobs = 0
                }
                return entry
            }
        }
        private func compilationTargets(key: String) throws -> [MPSGraphTensor] {
            if key.hasPrefix("forward-"), let index = Int(key.dropFirst("forward-".count)), frozenStages.indices.contains(index) {
                return frozenStages[index].map(\.source)
            }
            if key == "final-output" { return [output!] }
            if key == "final-loss" {
                try prepareLoss()
                return [output!, loss!, valueLoss!, gradientLoss!]
            }
            if key == "optimizer" {
                try prepareOptimizer(external: true)
                return optimizerOutputs.keys.sorted().map { optimizerOutputs[$0]! }
            }
            if key == "reverse-loss" {
                try prepareReverse(onlyJob: key)
                guard let lossReverse else { throw StudioError("Native compiler omitted loss derivatives.") }
                return lossReverse.derivatives
            }
            if key.hasPrefix("reverse-"), let index = Int(key.dropFirst("reverse-".count)) {
                try prepareReverse(onlyJob: key)
                guard let reverse = reverseStages[index] else { throw StudioError("Native compiler omitted stage derivatives.") }
                return reverse.derivatives
            }
            throw StudioError("Unknown native disposable compilation phase: \(key).")
        }
        private func executeValues(key: String, feeds: [MPSGraphTensor: MPSGraphTensorData], targets: [MPSGraphTensor],
                                   checkCancellation: () throws -> Void) throws -> [[Float]] {
            let results = try executeData(key: key, feeds: feeds, targets: targets, checkCancellation: checkCancellation)
            try packageCache?.stream.finish()
            try checkCancellation()
            return try results.map { try NativeGraphExecution.tensor($0).floatValues() }
        }
        private func immutableFeeds() throws -> [MPSGraphTensor: MPSGraphTensorData] {
            if let immutableData { return immutableData }
            var data: [MPSGraphTensor: MPSGraphTensorData] = [:]
            for (feed, value) in auxiliaryFeeds { data[feed] = try NativeGraphExecution.tensorData(value) }
            immutableData = data
            return data
        }
        struct Execution { let output: [Float]; let loss: Float?, valueLoss: Float?, gradientLoss: Float?; let updated: [String: NativeTensor]; let optimizerState: [String: NativeTensor]; let gradients: [String: NativeTensor] }
        func execute(rgb: [Float], adapters: [String: NativeTensor], reference: [Float]? = nil,
                     learningRate: Float? = nil, step: Int = 1, optimizerState: [String: NativeTensor] = [:],
                     gradientsOnly: Bool = false,
                     featureKey: String? = nil, checkCancellation: () throws -> Void = { try Task.checkCancellation() },
                     onStage: (Int, Int) -> Void = { _, _ in },
                     onOperation: (String, Int, Int) -> Void = { _, _, _ in }) throws -> Execution {
            try checkCancellation()
            return try autoreleasepool {
                try executePooled(rgb: rgb, adapters: adapters, reference: reference, learningRate: learningRate,
                    step: step, optimizerState: optimizerState, gradientsOnly: gradientsOnly, featureKey: featureKey,
                    checkCancellation: checkCancellation, onStage: onStage, onOperation: onOperation)
            }
        }
        private func executePooled(rgb: [Float], adapters: [String: NativeTensor], reference: [Float]?,
                     learningRate: Float?, step: Int, optimizerState: [String: NativeTensor], gradientsOnly: Bool, featureKey: String?,
                     checkCancellation: () throws -> Void, onStage: (Int, Int) -> Void,
                    onOperation: (String, Int, Int) -> Void) throws -> Execution {
            if let reference, stagesEnabled, learningRate != nil || gradientsOnly {
                return try executeCheckpointed(rgb: rgb, adapters: adapters, reference: reference,
                    learningRate: learningRate ?? 1, step: step, optimizerState: optimizerState, gradientsOnly: gradientsOnly,
                    featureKey: featureKey, checkCancellation: checkCancellation, onStage: onStage, onOperation: onOperation)
            }
            var feeds = try immutableFeeds()
            feeds[input] = try NativeGraphExecution.tensorData(.floats(rgb, shape: [1, 3, height, width]))
            for (name, placeholder) in parameterFeeds { feeds[placeholder] = try NativeGraphExecution.tensorData(adapters[name]!) }
            var targets = [output!]
            if let reference {
                try prepareLoss()
                guard reference.count == width * height * (target == "normal" ? 3 : 1), reference.allSatisfy(\.isFinite) else { throw StudioError("Native training target grid or values are invalid.") }
                feeds[targetInput!] = try NativeGraphExecution.tensorData(.floats(reference, shape: shape(output)))
                targets += [loss!, valueLoss!, gradientLoss!]
            }
            let optimize = reference != nil && learningRate != nil && !gradientsOnly
            let differentiate = reference != nil && gradientsOnly
            var ordered: [String] = []
            if optimize {
                try prepareOptimizer()
                guard let learningRate, learningRate.isFinite, learningRate > 0 else { throw StudioError("Invalid learning rate.") }
                feeds[self.learningRate!] = try NativeGraphExecution.tensorData(.floats([learningRate], shape: []))
                feeds[optimizerStep!] = try NativeGraphExecution.tensorData(.floats([Float(step)], shape: []))
                for (key, placeholder) in optimizerFeeds {
                    let parameterName = String(key.dropLast(2))
                    let value = optimizerState[key] ?? .floats([Float](repeating: 0, count: adapters[parameterName]!.shape.reduce(1, *)), shape: adapters[parameterName]!.shape)
                    feeds[placeholder] = try NativeGraphExecution.tensorData(value)
                }
                ordered = optimizerOutputs.keys.sorted()
                targets += ordered.map { optimizerOutputs[$0]! }
            }
            if differentiate {
                try prepareOptimizer()
                ordered = gradients.keys.sorted()
                targets += ordered.map { gradients[$0]! }
            }
            let finalFeeds = requiredFeeds(targets)
            let stageNeeds = frozenStages.map { requiredFeeds($0.map(\.source)) }
            let cacheKey = finalLayerOnly ? featureKey : nil
            if let cacheKey, let cached = featureCache[cacheKey] {
                onOperation("Reusing cached forward features", frozenStages.count, frozenStages.count)
                feeds.merge(cached) { _, new in new }
                featureOrder.removeAll { $0 == cacheKey }; featureOrder.append(cacheKey)
            } else {
                // Drop each intermediate immediately after its final consumer.
                // Keeping every previous stage feed recreates the RAM runaway.
                var remaining = finalFeeds
                var neededAfter = [Set<MPSGraphTensor>](repeating: [], count: frozenStages.count)
                for i in frozenStages.indices.reversed() { neededAfter[i] = remaining; remaining.formUnion(stageNeeds[i]) }
                for (i, stage) in frozenStages.enumerated() {
                    try checkCancellation()
                    onOperation("Forward pass", i, frozenStages.count)
                    let needed = stageNeeds[i]
                    let data = try autoreleasepool {
                        try executeData(key: "forward-\(i)", feeds: feeds.filter { needed.contains($0.key) }, targets: stage.map(\.source),
                                        checkCancellation: checkCancellation)
                    }
                    for (index, feature) in stage.enumerated() { feeds[feature.feed] = data[index] }
                    feeds = feeds.filter { neededAfter[i].contains($0.key) }
                    onStage(i + 1, frozenStages.count)
                    onOperation("Forward pass", i + 1, frozenStages.count)
                }
                if let cacheKey {
                    let frozen = Set(frozenStages.flatMap { $0.map(\.feed) })
                    let features = feeds.filter { frozen.contains($0.key) && finalFeeds.contains($0.key) }
                    let bytes = features.values.reduce(UInt64(0)) { $0 + UInt64($1.shape.reduce(4) { $0 * $1.intValue }) }
                    let budget = min(UInt64(2 * 1_073_741_824), MachineResources.current.maximumTrainingBytes / 16)
                    while featureBytes + bytes > budget, let oldest = featureOrder.first {
                        if let removed = featureCache.removeValue(forKey: oldest) {
                            featureBytes -= removed.values.reduce(UInt64(0)) { $0 + UInt64($1.shape.reduce(4) { $0 * $1.intValue }) }
                        }
                        featureOrder.removeFirst()
                    }
                    if bytes <= budget { featureCache[cacheKey] = features; featureOrder.append(cacheKey); featureBytes += bytes }
                }
            }
            try checkCancellation()
            onOperation(optimize ? "Computing loss and applying optimizer" : "Computing prediction and loss", 0, 1)
            let result = try executeValues(key: reference == nil ? "final-output" : "final-loss", feeds: feeds.filter { finalFeeds.contains($0.key) }, targets: targets,
                                           checkCancellation: checkCancellation)
            try checkCancellation()
            onOperation(optimize ? "Computing loss and applying optimizer" : "Computing prediction and loss", 1, 1)
            guard result[0].allSatisfy(\.isFinite), reference == nil || result[1][0].isFinite else { throw StudioError("Material model produced nonfinite values; weights remain untouched.") }
            var updated: [String: NativeTensor] = [:], state: [String: NativeTensor] = [:]
            if optimize {
                for (i, key) in ordered.enumerated() {
                    let values = result[i + 4]
                    guard values.allSatisfy(\.isFinite) else { throw StudioError("Native material gradients or optimizer state are nonfinite; stopped before applying weights.") }
                    if key.hasSuffix(".m") || key.hasSuffix(".v") { state[key] = .floats(values, shape: adapters[String(key.dropLast(2))]!.shape) }
                    else { updated[key] = .floats(values, shape: adapters[key]!.shape) }
                }
            }
            var derivative: [String: NativeTensor] = [:]
            if differentiate {
                for (index, name) in ordered.enumerated() {
                    let values = result[index + 4]
                    guard values.allSatisfy(\.isFinite) else { throw StudioError("Native adapter gradients are nonfinite.") }
                    derivative[name] = .floats(values, shape: adapters[name]!.shape)
                }
            }
            return Execution(output: result[0], loss: reference == nil ? nil : result[1][0], valueLoss: reference == nil ? nil : result[2][0], gradientLoss: reference == nil ? nil : result[3][0], updated: updated, optimizerState: state, gradients: derivative)
        }
        private struct ReverseStage {
            let key: String
            let seeds: [MPSGraphTensor]
            let inputs: [MPSGraphTensor]
            let derivatives: [MPSGraphTensor]
            let needs: Set<MPSGraphTensor>
        }
        // Explicit construction metadata for known same-shaped additions.
        // No operation-name inspection or source activation is needed for dy.
        private struct ImmutableLinearGelu {
            let input: MPSGraphTensor
            let preactivation: MPSGraphTensor
            let weights: MPSGraphTensor
        }
        private var immutableLinearGeluStages: [Int: ImmutableLinearGelu] = [:]
        private var geluStageInputs: [Int: MPSGraphTensor] = [:]
        private struct TailSliceResidual {
            let input: MPSGraphTensor
            let residual: MPSGraphTensor
            let start: Int
            let length: Int
        }
        private var tailSliceResiduals: [Int: TailSliceResidual] = [:]
        private var tailConcatOperands: [Int: [MPSGraphTensor]] = [:]
        private var sameShapeResidualOperands: [Int: [MPSGraphTensor]] = [:]
        private struct FrozenFusionResidual {
            let input: MPSGraphTensor
            let residualOperands: [MPSGraphTensor]
            let weights: MPSGraphTensor
            let descriptor: MPSGraphConvolution2DOpDescriptor
        }
        private var frozenFusionResiduals: [Int: FrozenFusionResidual] = [:]
        private var reverseStages: [Int: ReverseStage] = [:]
        private var lossReverse: ReverseStage?
        private var activeFeatures = Set<MPSGraphTensor>()
        private var activeStages = Set<Int>()
        private var producers: [MPSGraphTensor: Int] = [:]


        /// Reverse-mode differentiation at graph boundaries, with the complete
        /// native map loss. The VJP seed carries gradients through every chosen
        /// adapter, including residual/skip paths. Checkpoint tensors live in a
        /// bounded RAM pool; a missing checkpoint is recomputed from immutable
        /// pre-update weights. No activation files or swap-backed spill cache.
        private func prepareReverse(onlyJob: String? = nil) throws {
            if let onlyJob {
                if onlyJob == "reverse-loss", lossReverse != nil { return }
                if onlyJob.hasPrefix("reverse-"), let index = Int(onlyJob.dropFirst("reverse-".count)), reverseStages[index] != nil { return }
            } else if lossReverse != nil { return }
            try prepareLoss()
            let parameterSet = Set(parameterFeeds.values)
            for (index, stage) in frozenStages.enumerated() {
                for item in stage { producers[item.feed] = index }
                let needs = requiredFeeds(stage.map(\.source))
                if !needs.isDisjoint(with: parameterSet.union(activeFeatures)) {
                    activeStages.insert(index); activeFeatures.formUnion(stage.map(\.feed))
                }
            }
            let differentiable = parameterSet.union(activeFeatures)
            func reverse(_ sources: [MPSGraphTensor], key: String, scalarLoss: MPSGraphTensor? = nil) throws -> ReverseStage {
                let needs = requiredFeeds(sources)
                let inputs = needs.intersection(differentiable).sorted { $0.operation.name < $1.operation.name }
                var seeds: [MPSGraphTensor] = []
                var objective = scalarLoss
                if objective == nil {
                    var terms: [MPSGraphTensor] = []
                    for (index, source) in sources.enumerated() {
                        let seed = graph.placeholder(shape: source.shape, dataType: .float32, name: key + "_seed_\(index)")
                        seeds.append(seed)
                        terms.append(graph.reductionSum(with: mul(source, seed), axes: Array(0..<shape(source).count).ns, name: nil))
                    }
                    objective = terms.dropFirst().reduce(terms[0]) { add($0, $1) }
                }
                try Task.checkCancellation()
                if key.hasPrefix("reverse-"), let index = Int(key.dropFirst("reverse-".count)),
                   let tail = tailSliceResiduals[index], sources.count == 1, seeds.count == 1,
                   Set(inputs) == Set([tail.input, tail.residual]).intersection(differentiable),
                   needs.intersection(parameterSet).isEmpty,
                   shape(tail.input).count == 4, shape(sources[0]).count == 4,
                   shape(tail.residual) == shape(sources[0]),
                   shape(sources[0])[1] == tail.length, tail.start >= 0,
                   tail.start + tail.length <= shape(tail.input)[1],
                   [0, 2, 3].allSatisfy({ shape(tail.input)[$0] == shape(sources[0])[$0] }) {
                    // A fixed channel slice scatters dy into the original channels;
                    // the same-shaped initial-feature addition passes dy unchanged.
                    let derivatives = inputs.map { input -> MPSGraphTensor in
                        if input == tail.residual { return seeds[0] }
                        return graph.padTensor(seeds[0], with: .constant,
                            leftPadding: [0, tail.start, 0, 0].ns,
                            rightPadding: [0, shape(tail.input)[1] - tail.start - tail.length, 0, 0].ns,
                            constantValue: 0, name: nil)
                    }
                    return ReverseStage(key: key, seeds: seeds, inputs: inputs, derivatives: derivatives,
                        needs: requiredFeeds(derivatives).subtracting(Set(seeds)))
                }
                if key.hasPrefix("reverse-"), let index = Int(key.dropFirst("reverse-".count)),
                   let operands = tailConcatOperands[index], sources.count == 1, seeds.count == 1,
                   Set(operands).count == operands.count,
                   Set(inputs) == Set(operands).intersection(differentiable),
                   shape(sources[0]).count == 4,
                   operands.allSatisfy({ operand in shape(operand).count == 4 && [0, 2, 3].allSatisfy { axis in shape(operand)[axis] == shape(sources[0])[axis] } }),
                   operands.reduce(0, { $0 + shape($1)[1] }) == shape(sources[0])[1] {
                    // The channel concatenation VJP is one exact slice per input.
                    var results: [MPSGraphTensor: MPSGraphTensor] = [:], start = 0
                    for operand in operands {
                        let length = shape(operand)[1]
                        if inputs.contains(operand) { results[operand] = slice(seeds[0], 1, start, length) }
                        start += length
                    }
                    let derivatives = inputs.map { results[$0]! }
                    return ReverseStage(key: key, seeds: seeds, inputs: inputs, derivatives: derivatives,
                        needs: requiredFeeds(derivatives).subtracting(Set(seeds)))
                }
                if key.hasPrefix("reverse-"), let index = Int(key.dropFirst("reverse-".count)),
                   let input = geluStageInputs[index], sources.count == 1, seeds.count == 1,
                   inputs.count == 1, inputs[0] == input, shape(input) == shape(sources[0]),
                   needs.intersection(parameterSet).isEmpty {
                    // Reverse the exact erf GELU constructor: (z * 0.5) *
                    // (1 + erf(z / sqrt(2))). Every operand has the same shape.
                    let scaled = mul(input, c(1 / sqrt(2)))
                    let half = mul(input, c(0.5))
                    let first = mul(mul(seeds[0], add(c(1), graph.erf(with: scaled, name: nil))), c(0.5))
                    let gaussian = graph.exponent(with: sub(c(0), mul(scaled, scaled)), name: nil)
                    let erfDerivative = mul(c(2 / sqrt(Double.pi)), gaussian)
                    let second = mul(mul(mul(seeds[0], half), erfDerivative), c(1 / sqrt(2)))
                    let derivative = add(first, second)
                    return ReverseStage(key: key, seeds: seeds, inputs: inputs, derivatives: [derivative],
                        needs: requiredFeeds([derivative]).subtracting(Set(seeds)))
                }
                if key.hasPrefix("reverse-"), let index = Int(key.dropFirst("reverse-".count)),
                   let linear = immutableLinearGeluStages[index], sources.count == 1, seeds.count == 1,
                   inputs.count == 1, inputs[0] == linear.input, needs.intersection(parameterSet).isEmpty,
                   shape(linear.weights).count == 2, shape(linear.input).last == shape(linear.weights)[1],
                   shape(sources[0]).last == shape(linear.weights)[0],
                   shape(linear.preactivation) == shape(sources[0]),
                   Array(shape(linear.input).dropLast()) == Array(shape(sources[0]).dropLast()) {
                    // y = GELU(x W^T + frozen bias); dx = (dy GELU'(z)) W.
                    // Registration comes only from this exact forward constructor.
                    let scaled = mul(linear.preactivation, c(1 / sqrt(2)))
                    let half = mul(linear.preactivation, c(0.5))
                    let first = mul(mul(seeds[0], add(c(1), graph.erf(with: scaled, name: nil))), c(0.5))
                    let gaussian = graph.exponent(with: sub(c(0), mul(scaled, scaled)), name: nil)
                    let erfDerivative = mul(c(2 / sqrt(Double.pi)), gaussian)
                    let second = mul(mul(mul(seeds[0], half), erfDerivative), c(1 / sqrt(2)))
                    let derivative = graph.matrixMultiplication(primary: add(first, second), secondary: linear.weights, name: nil)
                    return ReverseStage(key: key, seeds: seeds, inputs: inputs, derivatives: [derivative],
                        needs: requiredFeeds([derivative]).subtracting(Set(seeds)))
                }
                if key.hasPrefix("reverse-"), let index = Int(key.dropFirst("reverse-".count)),
                   let operands = sameShapeResidualOperands[index], sources.count == 1, seeds.count == 1,
                   Set(operands).count == operands.count, Set(inputs) == Set(operands),
                   operands.allSatisfy({ shape($0) == shape(sources[0]) }) {
                    // d((a + frozenBias) + b)/da = dy and db = dy. Exact
                    // dimensions prove that neither derivative needs reduction.
                    return ReverseStage(key: key, seeds: seeds, inputs: inputs,
                        derivatives: inputs.map { _ in seeds[0] }, needs: [])
                }
                if key.hasPrefix("reverse-"), let index = Int(key.dropFirst("reverse-".count)),
                   let fusion = frozenFusionResiduals[index], sources.count == 1, seeds.count == 1,
                   Set(inputs) == Set([fusion.input] + fusion.residualOperands).intersection(differentiable),
                   needs.intersection(parameterSet).isEmpty {
                    // The immutable convolution's data derivative needs dy and
                    // weights. Concat's residual derivative slices dy channels.
                    // Static output dimensions replace primal shape scaffolding.
                    var results: [MPSGraphTensor: MPSGraphTensor] = [:]
                    if inputs.contains(fusion.input) {
                        results[fusion.input] = graph.convolution2DDataGradient(seeds[0], weights: fusion.weights,
                            outputShape: fusion.input.shape!, forwardConvolutionDescriptor: fusion.descriptor, name: nil)
                    }
                    var channel = 0
                    for operand in fusion.residualOperands {
                        let count = shape(operand)[1]
                        if inputs.contains(operand) {
                            let derivative = slice(seeds[0], 1, channel, count)
                            if let existing = results[operand] { results[operand] = add(existing, derivative) }
                            else { results[operand] = derivative }
                        }
                        channel += count
                    }
                    let ordered = try inputs.map { input -> MPSGraphTensor in
                        guard let result = results[input] else { throw StudioError("Frozen fusion lost an active derivative.") }
                        return result
                    }
                    return ReverseStage(key: key, seeds: seeds, inputs: inputs, derivatives: ordered,
                        needs: requiredFeeds(ordered).subtracting(Set(seeds)))
                }
                let derivatives = graph.gradients(of: objective!, with: inputs, name: "stage_vjp")
                let orderedDerivatives = try inputs.map { input -> MPSGraphTensor in
                    guard let derivative = derivatives[input] else { throw StudioError("Native stage lost a required training derivative.") }
                    return derivative
                }
                // Replay only values actually consumed by the derivative graph.
                // Inputs still describe every differentiated source dependency,
                // including constant derivatives that require no activation.
                let derivativeNeeds = requiredFeeds(orderedDerivatives).subtracting(Set(seeds))
                return ReverseStage(key: key, seeds: seeds, inputs: inputs,
                    derivatives: orderedDerivatives, needs: derivativeNeeds)
            }
            if onlyJob == nil || onlyJob == "reverse-loss" { lossReverse = try reverse([loss!], key: "reverse-loss", scalarLoss: loss) }
            for index in activeStages.sorted() where onlyJob == nil || onlyJob == "reverse-\(index)" {
                reverseStages[index] = try reverse(frozenStages[index].map(\.source), key: "reverse-\(index)")
            }
            if onlyJob == nil { try prepareOptimizer(external: true) }
        }
        private func sum(_ a: MPSGraphTensorData, _ b: MPSGraphTensorData, checkCancellation: () throws -> Void) throws -> MPSGraphTensorData {
            let key = "adjoint-sum-" + a.shape.map { String($0.intValue) }.joined(separator: ",")
            let graph = MPSGraph()
            let lhs = graph.placeholder(shape: a.shape, dataType: .float32, name: "lhs")
            let rhs = graph.placeholder(shape: a.shape, dataType: .float32, name: "rhs")
            let output = graph.addition(lhs, rhs, name: nil)
            if let packageCache {
                return try NativeGraphExecution.runPackaged(graph, feeds: [lhs: a, rhs: b], targets: [output], key: key, packages: packageCache, compile: {
                    try autoreleasepool {
                        let owner = MPSGraph()
                        let x = owner.placeholder(shape: a.shape, dataType: .float32, name: "lhs")
                        let y = owner.placeholder(shape: b.shape, dataType: .float32, name: "rhs")
                        let result = owner.addition(x, y, name: nil)
                        return try NativeGraphExecution.compilePackage(owner, feeds: [x: a, y: b], targets: [result], key: key, packages: packageCache)
                    }
                }, checkCancellation: checkCancellation)[0]
            }
            var cache: [String: MPSGraphExecutable] = [:]
            return try NativeGraphExecution.runData(graph, feeds: [lhs: a, rhs: b], targets: [output], cache: &cache,
                                                    checkCancellation: checkCancellation)[0]
        }
        /// Pick a fixed set of RAM checkpoints once per Program. A generator
        /// output checkpoint prevents every RRDB VJP from replaying the entire
        /// adapted generator. Remaining checkpoints minimize exact replay work
        /// per byte over the dependency DAG, within the same hard byte cap.
        private func checkpointAnchors(candidates: [MPSGraphTensor], budget: UInt64,
                    stageNeeds: [Set<MPSGraphTensor>],
                    frozenRoots: Set<MPSGraphTensor>, checkCancellation: () throws -> Void) throws -> Set<MPSGraphTensor> {
            func bytes(_ tensor: MPSGraphTensor) -> UInt64 { UInt64(shape(tensor).reduce(4, *)) }
            let byName = Dictionary(uniqueKeysWithValues: candidates.map { ($0.operation.name, $0) })
            if let checkpointNames {
                guard Set(checkpointNames).count == checkpointNames.count else { throw StudioError("Native checkpoint plan has duplicate boundaries.") }
                let restored = try checkpointNames.map { name -> MPSGraphTensor in
                    guard let tensor = byName[name] else { throw StudioError("Native checkpoint boundary changed: \(name).") }
                    return tensor
                }
                guard restored.reduce(UInt64(0), { $0 + bytes($1) }) <= budget else { throw StudioError("Native checkpoint plan exceeds its RAM limit.") }
                return Set(restored)
            }
            func save(_ anchors: Set<MPSGraphTensor>) -> Set<MPSGraphTensor> {
                checkpointNames = anchors.map { $0.operation.name }.sorted()
                return anchors
            }
            if budget == 0 { return save([]) }
            let totalBytes = candidates.reduce(UInt64(0)) { $0 + bytes($1) }
            if totalBytes <= budget { return save(Set(candidates)) }
            let interval = max(1, Int(totalBytes / budget + (totalBytes % budget == 0 ? 0 : 1)))
            var original = Set<MPSGraphTensor>(), originalBytes: UInt64 = 0
            for (index, feed) in candidates.enumerated() where index % interval == 0 {
                if originalBytes + bytes(feed) <= budget { original.insert(feed); originalBytes += bytes(feed) }
            }
            // Every current cut has one output. Preserve the previous safe
            // policy if future cuts expose multiple independent outputs.
            guard frozenStages.allSatisfy({ $0.count == 1 }) else { return save(original) }
            let integerNeeds = stageNeeds.map { needs in needs.compactMap { producers[$0] }.sorted() }
            // Score the dependencies actually consumed by each VJP. Using
            // the primal stage's inputs overvalues activations that constant
            // and explicit derivatives never replay (residuals, slices, GELU).
            let integerLossNeeds = lossReverse!.needs.compactMap { producers[$0] }.sorted()
            let integerReverseNeeds = reverseStages.mapValues { $0.needs.compactMap { producers[$0] }.sorted() }
            let frozenIndices = Set(frozenRoots.map { producers[$0]! })
            let reverseOrder = activeStages.sorted().reversed()
            func replayCount(_ anchors: Set<MPSGraphTensor>) -> Int {
                var available = [Bool](repeating: false, count: frozenStages.count)
                for feed in anchors { available[producers[feed]!] = true }
                for index in frozenIndices { available[index] = true }
                func replay(_ needs: [Int]) -> Int {
                    var visited = [Bool](repeating: false, count: frozenStages.count), count = 0
                    func collect(_ index: Int) {
                        if visited[index] || available[index] { return }
                        visited[index] = true; count += 1
                        for dependency in integerNeeds[index] { collect(dependency) }
                    }
                    for index in needs { collect(index) }
                    return count
                }
                var count = replay(integerLossNeeds)
                for index in reverseOrder {
                    count += replay(integerReverseNeeds[index]!)
                    available[index] = false
                }
                return count
            }
            var chosen = Set<MPSGraphTensor>(), chosenBytes: UInt64 = 0
            if let generatorAnchor, candidates.contains(generatorAnchor), bytes(generatorAnchor) <= budget {
                chosen.insert(generatorAnchor); chosenBytes += bytes(generatorAnchor)
            }
            while true {
                try checkCancellation()
                let previous = replayCount(chosen)
                var best: (feed: MPSGraphTensor, saved: Int, size: UInt64)?
                for feed in candidates {
                    let size = bytes(feed)
                    guard !chosen.contains(feed), chosenBytes + size <= budget else { continue }
                    var proposal = chosen; proposal.insert(feed)
                    let saved = previous - replayCount(proposal)
                    guard saved > 0 else { continue }
                    if best == nil || Double(saved) / Double(size) > Double(best!.saved) / Double(best!.size) { best = (feed, saved, size) }
                }
                guard let best else { break }
                chosen.insert(best.feed); chosenBytes += best.size
            }
            return save(replayCount(chosen) < replayCount(original) ? chosen : original)
        }
        private func executeCheckpointed(rgb: [Float], adapters: [String: NativeTensor], reference: [Float],
                    learningRate: Float, step: Int, optimizerState: [String: NativeTensor], gradientsOnly: Bool, featureKey: String?,
                    checkCancellation: () throws -> Void, onStage: (Int, Int) -> Void,
                    onOperation: (String, Int, Int) -> Void) throws -> Execution {
            guard learningRate.isFinite, learningRate > 0, reference.count == width * height * (target == "normal" ? 3 : 1),
                  reference.allSatisfy(\.isFinite) else { throw StudioError("Invalid native training target or learning rate.") }
            onOperation("Preparing backward graph", 0, 1)
            try prepareReverse()
            onOperation("Preparing backward graph", 1, 1)
            let stageNeeds = frozenStages.map { requiredFeeds($0.map(\.source)) }
            let finalTargets = [output!, loss!, valueLoss!, gradientLoss!]
            let finalNeeds = requiredFeeds(finalTargets)
            var base = try immutableFeeds()
            base[input] = try NativeGraphExecution.tensorData(.floats(rgb, shape: [1, 3, height, width]))
            base[targetInput!] = try NativeGraphExecution.tensorData(.floats(reference, shape: shape(output)))
            for (name, feed) in parameterFeeds { base[feed] = try NativeGraphExecution.tensorData(adapters[name]!) }
            let parameterNames = Dictionary(uniqueKeysWithValues: parameterFeeds.map { ($0.value, $0.key) })
            let boundarySet = Set(producers.keys)
            let backwardNeeds = activeStages.reduce(into: finalNeeds) { $0.formUnion(stageNeeds[$1]) }
            let frozenRoots = backwardNeeds.intersection(boundarySet).subtracting(activeFeatures)
            // Stage outputs already own compact logical buffers. Retain those
            // buffers for replay instead of synchronously reading whole-grid
            // activations into Data and uploading them again for every VJP.
            var checkpoints: [MPSGraphTensor: MPSGraphTensorData] = [:]
            var checkpointBytes: UInt64 = 0
            var frozenValues: [MPSGraphTensor: MPSGraphTensorData] = [:]
            // Keep cuts that reduce replay most within the fixed RAM budget.
            let candidates = frozenStages.flatMap { $0.map(\.feed) }.filter { activeFeatures.contains($0) && backwardNeeds.contains($0) }
            let checkpointBudget = min(checkpointByteLimit ?? UInt64(2 * 1_073_741_824), MachineResources.current.maximumTrainingBytes / 2)
            let anchorSet = try checkpointAnchors(candidates: candidates, budget: checkpointBudget, stageNeeds: stageNeeds,
                frozenRoots: frozenRoots, checkCancellation: checkCancellation)
            var feeds = base
            let frozenCacheHit: Bool
            if let featureKey, let cached = featureCache[featureKey], frozenRoots.isSubset(of: Set(cached.keys)) {
                frozenValues = cached; feeds.merge(cached) { _, new in new }; frozenCacheHit = true
                featureOrder.removeAll { $0 == featureKey }; featureOrder.append(featureKey)
            } else { frozenCacheHit = false }
            var remaining = finalNeeds
            var neededAfter = [Set<MPSGraphTensor>](repeating: [], count: frozenStages.count)
            for i in frozenStages.indices.reversed() { neededAfter[i] = remaining; remaining.formUnion(stageNeeds[i]) }
            for (index, stage) in frozenStages.enumerated() {
                try checkCancellation()
                onOperation("Forward pass", index, frozenStages.count)
                let data: [MPSGraphTensorData]
                if !activeStages.contains(index), frozenCacheHit {
                    // Cached frozen roots are the exact terminal features of
                    // the parameter-free prefix; no decoder/adapter is skipped.
                    onOperation("Forward pass (cached features)", index + 1, frozenStages.count)
                    continue
                } else {
                    data = try autoreleasepool {
                        try executeData(key: "forward-\(index)", feeds: feeds.filter { stageNeeds[index].contains($0.key) }, targets: stage.map(\.source),
                                        checkCancellation: checkCancellation)
                    }
                }
                for (offset, item) in stage.enumerated() {
                    feeds[item.feed] = data[offset]
                    if anchorSet.contains(item.feed) {
                        let checkpoint = data[offset]
                        let array = checkpoint.mpsndarray()
                        guard array.parent == nil, array.resourceSize() == shape(item.feed).reduce(4, *) else {
                            throw StudioError("Native training checkpoint retained noncompact activation storage.")
                        }
                        checkpoints[item.feed] = checkpoint
                        checkpointBytes += UInt64(array.resourceSize())
                        packageCache?.statistics.peakCheckpointBytes = max(packageCache?.statistics.peakCheckpointBytes ?? 0, checkpointBytes)
                    }
                    if frozenRoots.contains(item.feed) { frozenValues[item.feed] = data[offset] }
                }
                feeds = feeds.filter { neededAfter[index].contains($0.key) }
                onStage(index + 1, frozenStages.count)
                onOperation("Forward pass", index + 1, frozenStages.count)
            }
            if let featureKey, !frozenValues.isEmpty, featureCache[featureKey] == nil {
                let bytes = frozenValues.values.reduce(UInt64(0)) { $0 + UInt64($1.shape.reduce(4) { $0 * $1.intValue }) }
                let budget = min(UInt64(2 * 1_073_741_824), MachineResources.current.maximumTrainingBytes / 16)
                while featureBytes + bytes > budget, let oldest = featureOrder.first {
                    if let removed = featureCache.removeValue(forKey: oldest) {
                        featureBytes -= removed.values.reduce(UInt64(0)) { $0 + UInt64($1.shape.reduce(4) { $0 * $1.intValue }) }
                    }
                    featureOrder.removeFirst()
                }
                if bytes <= budget { featureCache[featureKey] = frozenValues; featureOrder.append(featureKey); featureBytes += bytes }
            }
            onOperation("Computing training loss", 0, 1)
            let result = try executeValues(key: "final-loss", feeds: feeds.filter { finalNeeds.contains($0.key) }, targets: finalTargets,
                                           checkCancellation: checkCancellation)
            guard result[0].allSatisfy(\.isFinite), result[1][0].isFinite else { throw StudioError("Material model produced nonfinite values.") }
            onOperation("Computing training loss", 1, 1)
            feeds.removeAll()
            var adjoints: [MPSGraphTensor: MPSGraphTensorData] = [:]
            var gradientBuffers: [String: MPSGraphTensorData] = [:]
            let reverseOrder = activeStages.sorted().reversed()
            var reversePosition = 0
            func replayInputs(_ needs: Set<MPSGraphTensor>) throws -> [MPSGraphTensor: MPSGraphTensorData] {
                var replay = Set<Int>(), visited = Set<MPSGraphTensor>()
                func collect(_ tensor: MPSGraphTensor) throws {
                    guard visited.insert(tensor).inserted else { return }
                    if base[tensor] != nil || frozenValues[tensor] != nil || checkpoints[tensor] != nil { return }
                    guard let index = producers[tensor] else { throw StudioError("Native checkpoint dependency is unavailable.") }
                    if replay.insert(index).inserted {
                        for dependency in stageNeeds[index] where boundarySet.contains(dependency) || base[dependency] != nil {
                            try collect(dependency)
                        }
                    }
                }
                let required = needs.filter { boundarySet.contains($0) || base[$0] != nil }
                for tensor in required { try collect(tensor) }
                let order = replay.sorted()
                var remaining = required
                var neededAfter = [Set<MPSGraphTensor>](repeating: [], count: order.count)
                for position in order.indices.reversed() {
                    neededAfter[position] = remaining
                    remaining.formUnion(stageNeeds[order[position]])
                }
                var live: [MPSGraphTensor: MPSGraphTensorData] = [:]
                func value(_ tensor: MPSGraphTensor) throws -> MPSGraphTensorData {
                    if let value = live[tensor] ?? base[tensor] ?? frozenValues[tensor] { return value }
                    if let checkpoint = checkpoints[tensor] { return checkpoint }
                    throw StudioError("Native replay lost a required boundary value.")
                }
                // Replay each ancestor once in forward order. A size-limited
                // recursive memo can forget a shared residual ancestor while
                // its siblings are being resolved and repeat whole prefixes.
                // Last-consumer release bounds storage without that repetition.
                for (position, index) in order.enumerated() {
                    try checkCancellation()
                    packageCache?.statistics.recomputedStages += 1
                    let backwardLabel = reversePosition == 0 ? "Loss gradients" : "Backward pass \(reversePosition)/\(reverseOrder.count)"
                    onOperation(backwardLabel + " · recomputing inputs", position, order.count)
                    var inputs: [MPSGraphTensor: MPSGraphTensorData] = [:]
                    for dependency in stageNeeds[index] where boundarySet.contains(dependency) || base[dependency] != nil {
                        inputs[dependency] = try value(dependency)
                    }
                    let data = try autoreleasepool {
                        try executeData(key: "forward-\(index)", feeds: inputs, targets: frozenStages[index].map(\.source),
                                        checkCancellation: checkCancellation)
                    }
                    for (offset, item) in frozenStages[index].enumerated() {
                        live[item.feed] = data[offset]
                        // Reverse traversal frees anchors as soon as their
                        // consumers finish. Reuse that same bounded capacity
                        // for replayed features needed by earlier VJPs instead
                        // of recomputing the same prefix for each derivative.
                        let bytes = UInt64(shape(item.feed).reduce(4, *))
                        if activeFeatures.contains(item.feed), backwardNeeds.contains(item.feed),
                           checkpoints[item.feed] == nil, checkpointBytes + bytes <= checkpointBudget {
                            checkpoints[item.feed] = data[offset]
                            checkpointBytes += bytes
                            packageCache?.statistics.peakCheckpointBytes = max(packageCache?.statistics.peakCheckpointBytes ?? 0, checkpointBytes)
                        }
                    }
                    live = live.filter { neededAfter[position].contains($0.key) }
                    onOperation(backwardLabel + " · recomputing inputs", position + 1, order.count)
                }
                var inputs: [MPSGraphTensor: MPSGraphTensorData] = [:]
                for tensor in required { inputs[tensor] = try value(tensor) }
                return inputs
            }
            func backward(_ plan: ReverseStage, seeds: [MPSGraphTensorData]) throws {
                try checkCancellation()
                var inputs = try replayInputs(plan.needs)
                for (index, seed) in seeds.enumerated() { inputs[plan.seeds[index]] = seed }
                let data = try autoreleasepool {
                    try executeData(key: plan.key, feeds: inputs, targets: plan.derivatives, checkCancellation: checkCancellation)
                }
                for (index, input) in plan.inputs.enumerated() {
                    if let name = parameterNames[input] {
                        // Reading each factor here serializes every reverse
                        // stage with the CPU. Accumulate on the GPU and read
                        // the compact factors once the backward pass finishes.
                        if let old = gradientBuffers[name] {
                            gradientBuffers[name] = try sum(old, data[index], checkCancellation: checkCancellation)
                        } else { gradientBuffers[name] = data[index] }
                    } else if let old = adjoints[input] { adjoints[input] = try sum(old, data[index], checkCancellation: checkCancellation) }
                    else { adjoints[input] = data[index] }
                }
            }
            onOperation("Computing loss gradients", 0, 1)
            try backward(lossReverse!, seeds: [])
            onOperation("Computing loss gradients", 1, 1)
            for index in reverseOrder {
                reversePosition += 1
                onOperation("Backward pass", reversePosition - 1, reverseOrder.count)
                let stage = frozenStages[index]
                let seeds = stage.map { adjoints[$0.feed] }
                if seeds.allSatisfy({ $0 == nil }) { continue }
                let completeSeeds = try stage.enumerated().map { offset, item in
                    try seeds[offset] ?? NativeGraphExecution.tensorData(.floats([Float](repeating: 0, count: shape(item.source).reduce(1, *)), shape: shape(item.source)))
                }
                for item in stage { adjoints.removeValue(forKey: item.feed) }
                try autoreleasepool { try backward(reverseStages[index]!, seeds: completeSeeds) }
                // Reverse consumers are finished; discarded anchors cannot be
                // needed by a later (earlier-in-forward-order) VJP.
                for item in stage {
                    if let removed = checkpoints.removeValue(forKey: item.feed) {
                        checkpointBytes -= UInt64(removed.shape.reduce(4) { $0 * $1.intValue })
                    }
                }
                onStage(frozenStages.count - index, frozenStages.count)
                onOperation("Backward pass", reversePosition, reverseOrder.count)
            }
            try packageCache?.stream.finish()
            try checkCancellation()
            let parameterGradients = try gradientBuffers.mapValues { try NativeGraphExecution.tensor($0) }
            for name in parameterFeeds.keys {
                guard let gradient = parameterGradients[name], try gradient.floatValues().allSatisfy(\.isFinite) else {
                    throw StudioError("Native training returned a missing or nonfinite adapter gradient.")
                }
                if !gradientsOnly { base[gradients[name]!] = try NativeGraphExecution.tensorData(gradient) }
            }
            if gradientsOnly {
                return Execution(output: result[0], loss: result[1][0], valueLoss: result[2][0], gradientLoss: result[3][0],
                    updated: [:], optimizerState: [:], gradients: parameterGradients)
            }
            base[self.learningRate!] = try NativeGraphExecution.tensorData(.floats([learningRate], shape: []))
            base[optimizerStep!] = try NativeGraphExecution.tensorData(.floats([Float(step)], shape: []))
            for (key, feed) in optimizerFeeds {
                let name = String(key.dropLast(2))
                let value = optimizerState[key] ?? .floats([Float](repeating: 0, count: adapters[name]!.shape.reduce(1, *)), shape: adapters[name]!.shape)
                base[feed] = try NativeGraphExecution.tensorData(value)
            }
            let ordered = optimizerOutputs.keys.sorted()
            let optimizerTargets = ordered.map { optimizerOutputs[$0]! }, optimizerNeeds = requiredFeeds(ordered.map { optimizerOutputs[$0]! })
            onOperation("Applying optimizer update", 0, 1)
            let updates = try executeValues(key: "optimizer", feeds: base.filter { optimizerNeeds.contains($0.key) }, targets: optimizerTargets,
                                            checkCancellation: checkCancellation)
            onOperation("Applying optimizer update", 1, 1)
            var updated: [String: NativeTensor] = [:], state: [String: NativeTensor] = [:]
            for (index, key) in ordered.enumerated() {
                guard updates[index].allSatisfy(\.isFinite) else { throw StudioError("Native optimizer returned nonfinite weights.") }
                let name = key.hasSuffix(".m") || key.hasSuffix(".v") ? String(key.dropLast(2)) : key
                let value = NativeTensor.floats(updates[index], shape: adapters[name]!.shape)
                if name == key { updated[key] = value } else { state[key] = value }
            }
            try checkCancellation()
            return Execution(output: result[0], loss: result[1][0], valueLoss: result[2][0], gradientLoss: result[3][0], updated: updated, optimizerState: state, gradients: [:])
        }
        private func prepareLoss() throws {
            if targetInput != nil { return }
            let reference = graph.placeholder(shape: output.shape, dataType: .float32, name: "native_linear_target")
            targetInput = reference
            let axes = [0, 1, 2, 3].ns
            let absolute = graph.mean(of: graph.absolute(with: sub(output, reference), name: nil), axes: axes, name: nil)
            var detail = c(0)
            for step in [1, 2, 4, 8] { for axis in [2, 3] {
                let length = shape(output)[axis] - step
                let pd = sub(slice(output, axis, step, length), slice(output, axis, 0, length))
                let td = sub(slice(reference, axis, step, length), slice(reference, axis, 0, length))
                let term = graph.mean(of: graph.absolute(with: sub(pd, td), name: nil), axes: axes, name: nil)
                detail = add(detail, div(term, c(Double(step))))
            } }
            valueLoss = absolute; gradientLoss = detail; loss = add(absolute, mul(detail, c(4)))
        }
        private func prepareOptimizer(external: Bool = false) throws {
            if optimizerPrepared { return }
            guard !parameterFeeds.isEmpty else { throw StudioError("Native training needs recorded material LoRA layers.") }
            try prepareLoss()
            let allParameters = parameterFeeds.keys.sorted()
            let derivative = external ? [:] : graph.gradients(of: loss!, with: allParameters.map { parameterFeeds[$0]! }, name: "material_lora_gradients")
            var normSquared = c(0)
            for key in allParameters {
                let gradient: MPSGraphTensor
                if external { gradient = graph.placeholder(shape: parameterFeeds[key]!.shape, dataType: .float32, name: key + ".gradient") }
                else {
                    guard let value = derivative[parameterFeeds[key]!] else { throw StudioError("Material graph could not differentiate adapter \(key).") }
                    gradient = value
                }
                gradients[key] = gradient
                normSquared = add(normSquared, graph.reductionSum(with: mul(gradient, gradient), axes: Array(0..<shape(gradient).count).ns, name: nil))
            }
            let clip = graph.minimum(c(1), div(c(1), add(graph.squareRoot(with: normSquared, name: nil), c(1e-6))), name: nil)
            learningRate = graph.placeholder(shape: [], dataType: .float32, name: "learning_rate")
            optimizerStep = graph.placeholder(shape: [], dataType: .float32, name: "optimizer_step")
            let b1 = c(0.9), b2 = c(0.999)
            let correction1 = sub(c(1), graph.power(b1, optimizerStep!, name: nil))
            let correction2 = sub(c(1), graph.power(b2, optimizerStep!, name: nil))
            for key in allParameters {
                let parameter = parameterFeeds[key]!, parameterShape = parameter.shape!
                let m = graph.placeholder(shape: parameterShape, dataType: .float32, name: key + ".m")
                let v = graph.placeholder(shape: parameterShape, dataType: .float32, name: key + ".v")
                optimizerFeeds[key + ".m"] = m; optimizerFeeds[key + ".v"] = v
                let gradient = mul(gradients[key]!, clip)
                let nextM = add(mul(m, b1), mul(gradient, c(0.1)))
                let nextV = add(mul(v, b2), mul(mul(gradient, gradient), c(0.001)))
                let numerator = div(nextM, correction1)
                let denominator = add(div(graph.squareRoot(with: nextV, name: nil), graph.squareRoot(with: correction2, name: nil)), c(1e-8))
                optimizerOutputs[key] = sub(parameter, mul(learningRate!, div(numerator, denominator)))
                optimizerOutputs[key + ".m"] = nextM; optimizerOutputs[key + ".v"] = nextV
            }
        }
    }
}

/// Disk packages contain only compiled code and value metadata. Loaded
/// executables live through GPU completion, then release their scratch arenas.
private final class NativeGraphPackageCache {
    var statistics = NativeGraphExecution.Statistics()
    let stream = NativeGraphExecution.Stream()
    struct Entry {
        let url: URL
        let inputNames: [String]
        let inputShapes: [[Int]]
        let inputTypes: [MPSDataType]
        let requestedIndices: [Int]
        let outputShapes: [[Int]]
        let outputTypes: [MPSDataType]
        let bytes: UInt64
        let persistent: Bool
    }
    private struct Metadata: Codable {
        let inputNames: [String]
        let inputShapes: [[Int]]
        let inputTypes: [UInt32]
        let requestedIndices: [Int]
        let outputShapes: [[Int]]
        let outputTypes: [UInt32]
        let bytes: UInt64
        var valid: Bool {
            func validShape(_ shape: [Int]) -> Bool {
                var count = 4
                for dimension in shape {
                    let product = count.multipliedReportingOverflow(by: dimension)
                    guard dimension > 0, !product.overflow else { return false }
                    count = product.partialValue
                }
                return true
            }
            let allowed = Set([MPSDataType.float32.rawValue, MPSDataType.int32.rawValue, MPSDataType.int64.rawValue])
            guard bytes > 0, bytes <= 1 << 30, !inputNames.contains(""), Set(inputNames).count == inputNames.count,
                  inputNames.count == inputShapes.count, inputNames.count == inputTypes.count,
                  inputTypes.allSatisfy({ allowed.contains($0) }), inputShapes.allSatisfy(validShape),
                  !requestedIndices.isEmpty, outputShapes.count == requestedIndices.count,
                  outputTypes.count == requestedIndices.count, outputShapes.allSatisfy(validShape),
                  outputTypes.allSatisfy({ $0 == MPSDataType.float32.rawValue }),
                  let last = requestedIndices.max(), last >= 0, last < requestedIndices.count,
                  Set(requestedIndices) == Set(0...last) else { return false }
            return true
        }
    }
    private let root: URL
    private let persistent: NativeGraphCodeCache?
    private var entries: [String: Entry] = [:]
    private var order: [String] = []
    private var bytes: UInt64 = 0
    private let maximumBytes: UInt64 = 1 << 30
    init(persistent: NativeGraphCodeCache? = nil) throws {
        self.persistent = persistent
        root = FileManager.default.temporaryDirectory.appendingPathComponent("NativeTrainingPrograms-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }
    deinit { try? FileManager.default.removeItem(at: root) }
    func entry(for key: String) -> Entry? {
        if let value = entries[key] {
            order.removeAll { $0 == key }; order.append(key)
            return value
        }
        guard let package = persistent?.package(for: key),
              let metadata = try? JSONDecoder().decode(Metadata.self, from: package.metadata), metadata.valid else { return nil }
        let inputTypes: [MPSDataType] = metadata.inputTypes.map {
            $0 == MPSDataType.int64.rawValue ? .int64 : $0 == MPSDataType.int32.rawValue ? .int32 : .float32
        }
        let entry = Entry(url: package.url, inputNames: metadata.inputNames, inputShapes: metadata.inputShapes,
            inputTypes: inputTypes, requestedIndices: metadata.requestedIndices,
            outputShapes: metadata.outputShapes, outputTypes: [MPSDataType](repeating: .float32, count: metadata.outputTypes.count),
            bytes: metadata.bytes, persistent: true)
        do { try remember(entry, key: key) } catch { return nil }
        statistics.diskPackageCacheHits += 1
        return entry
    }
    func packageURL() -> URL { root.appendingPathComponent(UUID().uuidString + ".mpsgraphpackage", isDirectory: true) }
    func insert(key: String, url: URL, inputNames: [String], inputShapes: [[Int]], inputTypes: [MPSDataType], requestedIndices: [Int], outputShapes: [[Int]], outputTypes: [MPSDataType]) throws -> Entry {
        guard let files = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey]) else {
            throw StudioError("Compiled native training program was not saved.")
        }
        var size: UInt64 = 0
        for case let file as URL in files {
            let values = try file.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            if values.isRegularFile == true { size += UInt64(values.fileSize ?? 0) }
        }
        guard size > 0, size <= maximumBytes else {
            try? FileManager.default.removeItem(at: url)
            throw StudioError("A compiled native training program exceeds its bounded code cache.")
        }
        let metadata = Metadata(inputNames: inputNames, inputShapes: inputShapes, inputTypes: inputTypes.map(\.rawValue),
            requestedIndices: requestedIndices, outputShapes: outputShapes, outputTypes: outputTypes.map(\.rawValue), bytes: size)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        let stored = (try? encoder.encode(metadata)).flatMap { persistent?.store(package: url, metadata: $0, key: key) }
        let entry = Entry(url: stored?.url ?? url, inputNames: inputNames, inputShapes: inputShapes, inputTypes: inputTypes,
            requestedIndices: requestedIndices, outputShapes: outputShapes, outputTypes: outputTypes, bytes: size, persistent: stored != nil)
        if stored != nil { try? FileManager.default.removeItem(at: url) }
        try remember(entry, key: key)
        return entry
    }
    private func remember(_ entry: Entry, key: String) throws {
        while bytes + entry.bytes > maximumBytes, let oldest = order.first {
            order.removeFirst()
            if let removed = entries.removeValue(forKey: oldest) {
                bytes -= removed.bytes
                if !removed.persistent { try FileManager.default.removeItem(at: removed.url) }
            }
        }
        entries[key] = entry; order.append(key); bytes += entry.bytes
    }
}

enum NativeGraphExecution {
    struct Statistics: Codable {
        var stages = 0
        var compilations = 0
        var diskPackageCacheHits = 0
        var compilationSeconds = 0.0
        var packageLoadSeconds = 0.0
        var executionSeconds = 0.0
        // MPS may commit intermediate buffers internally. This measures only
        // their final roots and must not be interpreted as GPU utilization.
        var finalCommandBufferGPUSeconds = 0.0
        var waitSeconds = 0.0
        var commandBuffers = 0
        var maximumInFlightStages = 0
        var recomputedStages = 0
        var peakCheckpointBytes: UInt64 = 0
    }

    /// A program-local, ordered GPU stream. The CPU can encode the next stage
    /// while the GPU executes its predecessor. A small submission window and
    /// conservative live-byte estimate bound overlap; large stages run alone.
    /// CPU reads, errors and cancellation always join the outstanding work.
    final class Stream {
        private final class Completion: @unchecked Sendable {
            private let lock = NSLock()
            private var failure: Error?
            func record(_ error: Error?) { lock.withLock { if failure == nil { failure = error } } }
            func check() throws { if let error = lock.withLock({ failure }) { throw error } }
        }
        private struct Submission {
            let commandBuffer: MPSCommandBuffer
            let finalBuffer: MTLCommandBuffer
            let executable: MPSGraphExecutable
            let inputs: [MPSGraphTensorData]
            let outputs: [MPSGraphTensorData]
            let completion: Completion
            let bytes: UInt64
        }
        fileprivate let queue = device?.makeCommandQueue()
        private var pending: [Submission] = []
        private var pendingBytes: UInt64 = 0
        private let byteLimit = min(UInt64(2 * 1_073_741_824), MachineResources.current.maximumTrainingBytes / 16)
        private(set) var finalCommandBufferGPUSeconds = 0.0
        private(set) var waitSeconds = 0.0
        private(set) var commandBuffers = 0
        private(set) var maximumInFlightStages = 0

        deinit { try? finish() }
        fileprivate func prepare(bytes: UInt64) throws {
            while let first = pending.first {
                let completed = first.finalBuffer.status == .completed || first.finalBuffer.status == .error
                guard completed || pending.count >= 2 || pendingBytes + bytes > byteLimit else { break }
                try finishFirst()
            }
        }
        private func finishFirst() throws {
            let submission = pending.removeFirst()
            pendingBytes -= submission.bytes
            let started = ProcessInfo.processInfo.systemUptime
            submission.finalBuffer.waitUntilCompleted()
            waitSeconds += ProcessInfo.processInfo.systemUptime - started
            finalCommandBufferGPUSeconds += max(0, submission.finalBuffer.gpuEndTime - submission.finalBuffer.gpuStartTime)
            // Keep the compiler arena and every input alive through completion.
            defer { withExtendedLifetime(submission) {} }
            if let error = submission.finalBuffer.error { throw error }
            try submission.completion.check()
        }
        func finish() throws {
            var failure: Error?
            while !pending.isEmpty {
                do { try finishFirst() } catch { if failure == nil { failure = error } }
            }
            if let failure { throw StudioError("Native GPU execution failed: \(failure.localizedDescription)") }
        }
        fileprivate func encode(_ executable: MPSGraphExecutable, inputs: [MPSGraphTensorData],
                                provided: [MPSGraphTensorData], buffers: [MTLBuffer],
                                shapes: [[Int]], types: [MPSDataType], bytes: UInt64) throws -> [MPSGraphTensorData] {
            guard let raw = queue?.makeCommandBuffer() else { throw StudioError("Could not create a native training command buffer.") }
            let commandBuffer = MPSCommandBuffer(commandBuffer: raw)
            let completion = Completion()
            let descriptor = MPSGraphExecutableExecutionDescriptor()
            descriptor.waitUntilCompleted = false
            descriptor.completionHandler = { _, error in completion.record(error) }
            executable.options = .synchronizeResults
            var submitted = false
            defer {
                if !submitted {
                    // MPS may commit intermediate command buffers while
                    // encoding. Commit/join its final buffer on every exit.
                    let last = commandBuffer.rootCommandBuffer
                    commandBuffer.commit(); last.waitUntilCompleted()
                }
            }
            let results = executable.encode(to: commandBuffer, inputs: inputs, results: provided, executionDescriptor: descriptor)
            guard results.count == provided.count else { throw StudioError("Native graph returned incomplete output storage.") }
            var independent = results
            for (index, result) in results.enumerated() {
                guard result.shape.map(\.intValue) == shapes[index], result.dataType == types[index] else {
                    throw StudioError("Native graph changed compact output shape or precision.")
                }
                if result === provided[index], result.mpsndarray().parent == nil { continue }
                result.mpsndarray().exportData(with: commandBuffer, to: buffers[index], destinationDataType: .float32,
                    offset: 0, rowStrides: nil)
                independent[index] = provided[index]
            }
            // Capture the final root after encode: MPS is allowed to commit
            // and replace the original root while encoding a large graph.
            let last = commandBuffer.rootCommandBuffer
            commandBuffer.commit()
            submitted = true
            pending.append(Submission(commandBuffer: commandBuffer, finalBuffer: last, executable: executable,
                inputs: inputs, outputs: independent, completion: completion, bytes: bytes))
            pendingBytes += bytes
            commandBuffers += 1
            maximumInFlightStages = max(maximumInFlightStages, pending.count)
            return independent
        }
    }
    // A requested value may serve several derivatives. Compile its storage
    // once, then reconstruct each requested slot from the stable index map.
    private static func uniqueTargets(_ targets: [MPSGraphTensor]) -> [MPSGraphTensor] {
        var seen = Set<MPSGraphTensor>()
        return targets.filter { seen.insert($0).inserted }
    }
    private static let device = MTLCreateSystemDefaultDevice()
    private static let queue = device?.makeCommandQueue()
    static func nearestNeighbor2(_ value: MPSGraphTensor, graph: MPSGraph) -> MPSGraphTensor {
        let shape = value.shape!.map(\.intValue)
        let expanded = graph.reshape(value, shape: [shape[0], shape[1], shape[2], 1, shape[3], 1].ns, name: nil)
        // MPSGraphTileOp has no autodiff implementation. Concatenating the
        // singleton axes repeats the exact same nearest-neighbor samples,
        // while its derivative sums both copies back into the source.
        let rows = graph.concatTensors([expanded, expanded], dimension: 3, name: nil)
        let samples = graph.concatTensors([rows, rows], dimension: 5, name: nil)
        return graph.reshape(samples, shape: [shape[0], shape[1], shape[2] * 2, shape[3] * 2].ns, name: nil)
    }
    static func tensorData(_ value: NativeTensor) throws -> MPSGraphTensorData {
        guard let device else { throw StudioError("Metal is unavailable for native material inference.") }
        let dtype: MPSDataType = value.dtype == "I64" ? .int64 : value.dtype == "I32" ? .int32 : .float32
        return MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device), data: value.bytes, shape: value.shape.ns, dataType: dtype)
    }
    static func run(_ graph: MPSGraph, feeds: [MPSGraphTensor: MPSGraphTensorData], targets: [MPSGraphTensor]) throws -> [[Float]] {
        var cache: [String: MPSGraphExecutable] = [:]
        return try run(graph, feeds: feeds, targets: targets, cache: &cache)
    }
    static func run(_ graph: MPSGraph, feeds: [MPSGraphTensor: MPSGraphTensorData], targets: [MPSGraphTensor], cache: inout [String: MPSGraphExecutable]) throws -> [[Float]] {
        let result = try runData(graph, feeds: feeds, targets: targets, cache: &cache)
        return result.map { value in
            let count = value.shape.reduce(1) { $0 * $1.intValue }
            var values = [Float](repeating: 0, count: count)
            values.withUnsafeMutableBytes { value.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
            return values
        }
    }
    static func tensor(_ value: MPSGraphTensorData) throws -> NativeTensor {
        let shape = value.shape.map(\.intValue)
        var bytes = Data(count: shape.reduce(4, *))
        bytes.withUnsafeMutableBytes { value.mpsndarray().readBytes($0.baseAddress!, strideBytes: nil) }
        return NativeTensor(dtype: "F32", shape: shape, bytes: bytes)
    }
    /// Loads bounded compiled code by an explicit phase/stage identity. Feeds
    /// are rebound only by unique explicit placeholder names from a cold graph.
    fileprivate static func runPackaged(_ graph: MPSGraph, feeds: [MPSGraphTensor: MPSGraphTensorData], targets: [MPSGraphTensor], key: String, packages: NativeGraphPackageCache,
                                       compile: (() throws -> NativeGraphPackageCache.Entry)? = nil,
                                       checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> [MPSGraphTensorData] {
        try Task.checkCancellation()
        try checkCancellation()
        guard let device, let queue else { throw StudioError("Metal is unavailable for native material computation.") }
        var named: [String: MPSGraphTensorData] = [:]
        for (tensor, data) in feeds {
            let name = tensor.operation.name
            guard !name.isEmpty, named.updateValue(data, forKey: name) == nil,
                  tensor.shape?.map(\.intValue) == data.shape.map(\.intValue), tensor.dataType == data.dataType else {
                throw StudioError("Native compiled program feeds lack unique, exact placeholder identities.")
            }
        }
        let entry: NativeGraphPackageCache.Entry
        if let existing = packages.entry(for: key) { entry = existing }
        else {
            // A cold compiler can allocate its own large workspace. Join the
            // preceding stage before allowing those two arenas to overlap.
            // Cached code retains asynchronous steady-state submission.
            try packages.stream.finish()
            try checkCancellation()
            let started = ProcessInfo.processInfo.systemUptime
            if let compile { entry = try compile() }
            else { entry = try autoreleasepool { try compilePackage(graph, feeds: feeds, targets: targets, key: key, packages: packages) } }
            packages.statistics.compilations += 1
            packages.statistics.compilationSeconds += ProcessInfo.processInfo.systemUptime - started
        }
        // A synchronous compiler cannot be interrupted. Honor Abort before
        // its completed package can start another allocation or GPU run.
        try Task.checkCancellation()
        try checkCancellation()
        guard targets.count == entry.requestedIndices.count,
              targets.map({ $0.shape!.map(\.intValue) }) == entry.outputShapes,
              targets.map(\.dataType) == entry.outputTypes else {
            throw StudioError("Native compiled program result metadata changed for \(key).")
        }
        let inputs = try entry.inputNames.enumerated().map { index, name -> MPSGraphTensorData in
            guard let data = named[name], data.shape.map(\.intValue) == entry.inputShapes[index], data.dataType == entry.inputTypes[index] else {
                throw StudioError("Native compiled program input metadata changed for \(key): \(name).")
            }
            return data
        }
        let inputBytes = inputs.reduce(UInt64(0)) { $0 + UInt64($1.shape.reduce(4) { $0 * $1.intValue }) }
        let outputBytes = entry.outputShapes.reduce(UInt64(0)) { $0 + UInt64($1.reduce(4, *)) }
        let reservation = max(UInt64(64 * 1_048_576), (inputBytes + outputBytes) * 4)
        try packages.stream.prepare(bytes: reservation)
        guard UInt64(device.currentAllocatedSize) <= MachineResources.current.maximumTrainingBytes else {
            throw StudioError("Native training exceeded this Mac's safe Metal working budget; stopped before another stage allocation.")
        }
        let loadStarted = ProcessInfo.processInfo.systemUptime
        let executable = MPSGraphExecutable(package: entry.url, descriptor: compilationDescriptor())
        packages.statistics.packageLoadSeconds += ProcessInfo.processInfo.systemUptime - loadStarted
        try Task.checkCancellation()
        try checkCancellation()
        // Package executables need not expose tensor identities. Reconstruct
        // every compiled output slot from the saved requested-index mapping.
        guard let lastIndex = entry.requestedIndices.max(), lastIndex >= 0, lastIndex < entry.requestedIndices.count else {
            throw StudioError("Native compiled program has invalid result order for \(key).")
        }
        var outputShapes = [[Int]?](repeating: nil, count: lastIndex + 1)
        var outputTypes = [MPSDataType?](repeating: nil, count: lastIndex + 1)
        for (index, compiledIndex) in entry.requestedIndices.enumerated() {
            guard outputShapes.indices.contains(compiledIndex) else { throw StudioError("Native compiled program has invalid result order for \(key).") }
            if let existing = outputShapes[compiledIndex] {
                guard existing == entry.outputShapes[index], outputTypes[compiledIndex] == entry.outputTypes[index] else {
                    throw StudioError("Native compiled program has conflicting result metadata for \(key).")
                }
            }
            outputShapes[compiledIndex] = entry.outputShapes[index]; outputTypes[compiledIndex] = entry.outputTypes[index]
        }
        guard outputShapes.allSatisfy({ $0 != nil }), outputTypes.allSatisfy({ $0 != nil }) else {
            throw StudioError("Native compiled program omitted output storage metadata for \(key).")
        }
        let executionStarted = ProcessInfo.processInfo.systemUptime
        let results = try runWithCompactOutputs(executable, inputs: inputs, shapes: outputShapes.map { $0! },
                                                types: outputTypes.map { $0! }, device: device, queue: queue,
                                                stream: packages.stream,
                                                checkCancellation: checkCancellation)
        packages.statistics.stages += 1
        packages.statistics.executionSeconds += ProcessInfo.processInfo.systemUptime - executionStarted
        let requested = try entry.requestedIndices.enumerated().map { index, compiledIndex -> MPSGraphTensorData in
            guard results.indices.contains(compiledIndex) else { throw StudioError("Native compiled program returned incomplete results.") }
            let result = results[compiledIndex]
            guard result.shape.map(\.intValue) == entry.outputShapes[index], result.dataType == entry.outputTypes[index] else {
                throw StudioError("Native compiled program changed result storage for \(key).")
            }
            return result
        }
        return requested
    }
    fileprivate static func compilePackage(_ graph: MPSGraph, feeds: [MPSGraphTensor: MPSGraphTensorData], targets: [MPSGraphTensor], key: String, packages: NativeGraphPackageCache) throws -> NativeGraphPackageCache.Entry {
        try Task.checkCancellation()
        guard let device else { throw StudioError("Metal is unavailable for native material computation.") }
        let descriptor = compilationDescriptor()
        let shaped = feeds.mapValues { MPSGraphShapedType(shape: $0.shape, dataType: $0.dataType) }
        let original = graph.compile(with: MPSGraphDevice(mtlDevice: device), feeds: shaped,
            targetTensors: uniqueTargets(targets), targetOperations: nil, compilationDescriptor: descriptor)
        try Task.checkCancellation()
        guard let inputs = original.feedTensors, let compiledTargets = original.targetTensors,
              Set(inputs.map { $0.operation.name }).count == inputs.count else {
            throw StudioError("Native compiled program has ambiguous input or output identities.")
        }
        let indices = try targets.map { tensor -> Int in
            guard let index = compiledTargets.firstIndex(of: tensor) else {
                throw StudioError("Native compiled program omitted a requested result.")
            }
            return index
        }
        guard Set(indices) == Set(compiledTargets.indices) else {
            throw StudioError("Native compiled program has unregistered output storage for \(key).")
        }
        let url = packages.packageURL()
        original.serialize(package: url, descriptor: nil)
        return try packages.insert(key: key, url: url,
            inputNames: inputs.map { $0.operation.name }, inputShapes: inputs.map { $0.shape!.map(\.intValue) }, inputTypes: inputs.map(\.dataType),
            requestedIndices: indices, outputShapes: targets.map { $0.shape!.map(\.intValue) }, outputTypes: targets.map(\.dataType))
    }
    private static func compilationDescriptor() -> MPSGraphCompilationDescriptor {
        let descriptor = MPSGraphCompilationDescriptor()
        descriptor.reducedPrecisionFastMath = .none
        descriptor.optimizationLevel = .level0
        // Keep public tensors in NCHW while permitting the GPU compiler to
        // choose channels-last convolution kernels on supported systems.
        if #available(macOS 26.4, *) { descriptor.convertLayoutToNHWC() }
        descriptor.waitForCompilationCompletion = true
        return descriptor
    }
    /// Execute directly into caller-owned logical buffers. Default outputs
    /// can retain graph scratch storage; exporting them afterward duplicates
    /// every whole-grid result while that arena is still alive.
    private static func runWithCompactOutputs(_ executable: MPSGraphExecutable, inputs: [MPSGraphTensorData],
                                              shapes: [[Int]], types: [MPSDataType], device: MTLDevice,
                                              queue: MTLCommandQueue, stream: Stream? = nil,
                                              checkCancellation: () throws -> Void) throws -> [MPSGraphTensorData] {
        try Task.checkCancellation()
        try checkCancellation()
        guard shapes.count == types.count else { throw StudioError("Native graph output metadata is incomplete.") }
        let inputBytes = inputs.reduce(UInt64(0)) { $0 + UInt64($1.shape.reduce(4) { $0 * $1.intValue }) }
        let outputBytes = shapes.reduce(UInt64(0)) { $0 + UInt64($1.reduce(4, *)) }
        let overlapBytes = max(UInt64(64 * 1_048_576), (inputBytes + outputBytes) * 4)
        try stream?.prepare(bytes: overlapBytes)
        let buffers = try zip(shapes, types).map { shape, type -> MTLBuffer in
            try checkCancellation()
            guard type == .float32 else { throw StudioError("Native graph results must remain Float32.") }
            let bytes = try shape.reduce(4) { count, dimension in
                let product = count.multipliedReportingOverflow(by: dimension)
                guard dimension > 0, !product.overflow else { throw StudioError("Native graph output dimensions exceed addressable storage.") }
                return product.partialValue
            }
            guard let buffer = device.makeBuffer(length: bytes, options: .storageModeShared) else {
                throw StudioError("Could not allocate compact native graph output storage.")
            }
            return buffer
        }
        let provided = buffers.enumerated().map { MPSGraphTensorData($0.element, shape: shapes[$0.offset].ns, dataType: types[$0.offset]) }
        if let stream {
            try checkCancellation()
            return try stream.encode(executable, inputs: inputs, provided: provided, buffers: buffers,
                shapes: shapes, types: types, bytes: overlapBytes)
        }
        let descriptor = MPSGraphExecutableExecutionDescriptor()
        descriptor.waitUntilCompleted = true
        executable.options = .synchronizeResults
        try Task.checkCancellation()
        try checkCancellation()
        let results = executable.run(with: queue, inputs: inputs, results: provided, executionDescriptor: descriptor)
        try Task.checkCancellation()
        try checkCancellation()
        guard results.count == provided.count else { throw StudioError("Native graph returned incomplete output storage.") }
        var independent = results
        var copy: MTLCommandBuffer?
        for (index, result) in results.enumerated() {
            try checkCancellation()
            guard result.shape.map(\.intValue) == shapes[index], result.dataType == types[index] else {
                throw StudioError("Native graph changed compact output shape or precision.")
            }
            if result === provided[index], result.mpsndarray().parent == nil { continue }
            // Some compiled read-only views return an input alias instead of
            // the supplied result. Compact only those exceptional slots into
            // their already allocated logical buffers, preserving strides.
            if copy == nil { copy = queue.makeCommandBuffer() }
            guard let copy else { throw StudioError("Could not copy an aliased native graph output.") }
            result.mpsndarray().exportData(with: copy, to: buffers[index], destinationDataType: .float32, offset: 0, rowStrides: nil)
            independent[index] = MPSGraphTensorData(buffers[index], shape: shapes[index].ns, dataType: types[index])
        }
        if let copy {
            copy.commit(); copy.waitUntilCompleted()
            if let error = copy.error { throw StudioError("Native graph output copy failed: \(error.localizedDescription)") }
        }
        try Task.checkCancellation()
        try checkCancellation()
        guard independent.allSatisfy({ $0.mpsndarray().parent == nil }) else {
            throw StudioError("Native graph retained aliased compact output storage.")
        }
        return independent
    }
    /// Returns compact, independent buffers: MPSGraph results can be views of
    /// its entire scratch arena, which must not survive with a boundary tensor.
    static func runData(_ graph: MPSGraph, feeds: [MPSGraphTensor: MPSGraphTensorData], targets: [MPSGraphTensor], cache: inout [String: MPSGraphExecutable],
                        stream: Stream? = nil,
                        checkCancellation: () throws -> Void = { try Task.checkCancellation() }) throws -> [MPSGraphTensorData] {
        try Task.checkCancellation()
        try checkCancellation()
        guard let device, let queue else { throw StudioError("Metal is unavailable for native material computation.") }
        let key = targets.map { String(describing: ObjectIdentifier($0)) }.joined(separator: "|")
        let executable: MPSGraphExecutable
        if let existing = cache[key] { executable = existing }
        else {
            // Each executable retains a Metal scratch arena. Keeping one for
            // every whole-grid block consumes >100 GiB at just 1K. Compact
            // mask feeds make recompilation cheap; retain only the current
            // stage so arenas can be freed before compiling the next stage.
            cache.removeAll(keepingCapacity: true)
            let descriptor = MPSGraphCompilationDescriptor()
            descriptor.reducedPrecisionFastMath = .none
            // Metal-only Float32 execution. Level1 also tries ANE placement
            // for tiny adapter/optimizer subgraphs, which cannot accept FP32.
            descriptor.optimizationLevel = .level0
            descriptor.waitForCompilationCompletion = true
            let shaped = Dictionary(uniqueKeysWithValues: feeds.map { ($0.key, MPSGraphShapedType(shape: $0.value.shape, dataType: $0.value.dataType)) })
            executable = graph.compile(with: MPSGraphDevice(mtlDevice: device), feeds: shaped, targetTensors: uniqueTargets(targets), targetOperations: nil, compilationDescriptor: descriptor)
            cache[key] = executable
        }
        try Task.checkCancellation()
        try checkCancellation()
        guard UInt64(device.currentAllocatedSize) <= MachineResources.current.maximumTrainingBytes else {
            cache.removeAll()
            throw StudioError("Native training exceeded this Mac's safe Metal working budget; stopped before another stage allocation.")
        }
        guard let feedTensors = executable.feedTensors, let compiledTargets = executable.targetTensors,
              compiledTargets.allSatisfy({ $0.shape != nil }) else {
            throw StudioError("Native material graph returned incomplete result identities.")
        }
        let executionInputs = try feedTensors.map { tensor -> MPSGraphTensorData in
            guard let value = feeds[tensor] else { throw StudioError("Native material graph omitted a required input.") }
            return value
        }
        let result = try runWithCompactOutputs(executable, inputs: executionInputs, shapes: compiledTargets.map { $0.shape!.map(\.intValue) },
                                               types: compiledTargets.map(\.dataType), device: device, queue: queue,
                                               stream: stream,
                                               checkCancellation: checkCancellation)
        var byTensor: [MPSGraphTensor: MPSGraphTensorData] = [:]
        for (index, tensor) in compiledTargets.enumerated() { byTensor[tensor] = result[index] }
        let requested = try targets.map { tensor -> MPSGraphTensorData in
            guard let value = byTensor[tensor] else { throw StudioError("Native material graph omitted a requested result.") }
            return value
        }
        return requested
    }
}

struct NativeMaterialRandom {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9e3779b97f4a7c15
        var value = state; value = (value ^ (value >> 30)) &* 0xbf58476d1ce4e5b9
        value = (value ^ (value >> 27)) &* 0x94d049bb133111eb
        return value ^ (value >> 31)
    }
    mutating func unit() -> Float { Float(next() >> 40) / Float(1 << 24) }
    mutating func shuffle<T>(_ values: inout [T]) {
        if values.count < 2 { return }
        for index in stride(from: values.count - 1, through: 1, by: -1) { values.swapAt(index, Int(next() % UInt64(index + 1))) }
    }
}

extension NativeTensor {
    static func floats(_ values: [Float], shape: [Int]) -> NativeTensor {
        NativeTensor(dtype: "F32", shape: shape, bytes: values.withUnsafeBytes { Data($0) })
    }
    func floatValues() throws -> [Float] {
        guard dtype == "F32", bytes.count % 4 == 0 else { throw StudioError("Material computation requires exact Float32 samples.") }
        return bytes.withUnsafeBytes { raw in (0..<bytes.count / 4).map { raw.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) } }
    }
}
private extension Array where Element == Int { var ns: [NSNumber] { map { NSNumber(value: $0) } } }

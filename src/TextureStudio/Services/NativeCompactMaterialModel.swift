import CoreFoundation
import CryptoKit
import Foundation
import MetalPerformanceShadersGraph

// NAF-style block design adapted from megvii-research/NAFNet (MIT).
// Copyright (c) 2022 megvii-model. The retained notice is NAFNet_LICENSE.
// This native graph uses a smaller task-specific topology and no upstream weights.

/// A small, fully trainable native material network. Its scalar and normal
/// families have distinct identities and only the requested output head.
/// PBRnxt weights and LoRA factors are never accepted by this backend.
final class NativeCompactMaterialModel: @unchecked Sendable {
    static let schema = "texture-studio-compact-material-v1"
    static let scalarArchitecture = "texture-studio-compact-scalar-native-v1"
    static let normalArchitecture = "texture-studio-compact-normal-native-v1"
    let target: String, networkWidth: Int, outputChannels: Int
    let architectureID: String, baseSHA256: String, initializationSeed: UInt64
    private(set) var weights: [String: NativeTensor]
    var configuration: [String: Any]
    private var cached: Program?

    static func architecture(for target: String) throws -> String {
        guard ["height", "roughness", "normal"].contains(target) else {
            throw StudioError("Choose height, roughness or normal for the compact material model.")
        }
        return target == "normal" ? normalArchitecture : scalarArchitecture
    }

    init(target: String, width: Int = 32, seed: UInt64 = 17) throws {
        architectureID = try Self.architecture(for: target)
        let initial = try Self.initialWeights(target: target, width: width, seed: seed)
        self.target = target; networkWidth = width; outputChannels = target == "normal" ? 3 : 1
        initializationSeed = seed; weights = initial
        baseSHA256 = try Self.weightIdentity(initial)
        configuration = ["schema": Self.schema, "architecture": architectureID,
            "model_family": target == "normal" ? "compact-normal" : "compact-scalar",
            "from_scratch": true, "target": target, "network_width": width,
            "input_channels": 3, "output_channels": outputChannels,
            "initialization_seed": seed, "initial_weights_sha256": baseSHA256,
            "training_scope": "all-weights", "scope": "full-model", "head_activation": "linear",
            "normal_encoding": "declared dataset convention; numeric source codes / code maximum",
            "parameter_count": initial.values.reduce(0) { $0 + $1.shape.reduce(1, *) },
            "runtime": "Apple MPSGraph Float32", "reduced_precision_fast_math": false]
    }

    static func load(checkpointURL: URL?, expectedSHA256: String? = nil, target: String,
                     width: Int = 32, seed: UInt64 = 17) throws -> NativeCompactMaterialModel {
        guard let checkpointURL else { return try .init(target: target, width: width, seed: seed) }
        let source = NativeMaterialModel.resolvedCheckpoint(checkpointURL)
        let snapshot = try NativeSafetensors(contentsOf: source, expectedSHA256: expectedSHA256)
        _ = try NativeMaterialCheckpoint.inspect(at: source, expectedSHA256: snapshot.sha256)
        guard let metadata = snapshot.metadata["configuration"] else {
            throw NativeCheckpointError.invalid("compact checkpoint has no configuration")
        }
        let configuration = try NativeMaterialTransfer.object(Data(metadata.utf8))
        try validateCheckpoint(snapshot, configuration: configuration)
        guard configuration["target"] as? String == target,
              let storedWidth = integer(configuration["network_width"]),
              storedWidth == width,
              let storedSeed = unsigned(configuration["initialization_seed"]) else {
            throw NativeCheckpointError.invalid("compact checkpoint target or network width differs from the selected model")
        }
        let model = try NativeCompactMaterialModel(target: target, width: storedWidth, seed: storedSeed)
        try model.updateWeights(snapshot.nativeTensors())
        model.configuration = configuration
        return model
    }

    static func validateCheckpoint(_ snapshot: NativeSafetensors, configuration: [String: Any]) throws {
        guard configuration["schema"] as? String == schema,
              let target = configuration["target"] as? String,
              let width = integer(configuration["network_width"]),
              let seed = unsigned(configuration["initialization_seed"]),
              configuration["architecture"] as? String == (try architecture(for: target)),
              configuration["model_family"] as? String == (target == "normal" ? "compact-normal" : "compact-scalar"),
              boolean(configuration["from_scratch"]) == true,
              integer(configuration["input_channels"]) == 3,
              integer(configuration["output_channels"]) == (target == "normal" ? 3 : 1),
              configuration["training_scope"] as? String == "all-weights",
              configuration["scope"] as? String == "full-model",
              configuration["head_activation"] as? String == "linear",
              configuration["runtime"] as? String == "Apple MPSGraph Float32",
              configuration["native_runtime"] as? String == "Apple MPSGraph Float32, reduced precision fast math disabled",
              boolean(configuration["reduced_precision_fast_math"]) == false,
              boolean(configuration["image_padding"]) == false,
              boolean(configuration["image_resizing"]) == false,
              let step = integer(configuration["step"]), step >= 0,
              let size = integer(configuration["training_size"]), validGrid(width: size, height: size),
              configuration["base"] == nil, configuration["layers"] == nil else {
            throw NativeCheckpointError.invalid("compact model identity or output family is invalid")
        }
        let shapes = try weightShapes(target: target, width: width)
        guard Set(snapshot.tensors.keys) == Set(shapes.keys), snapshot.tensors.allSatisfy({
            $0.value.dtype == "F32" && $0.value.shape == shapes[$0.key]
        }) else { throw NativeCheckpointError.invalid("compact checkpoint tensors differ from its complete architecture") }
        try snapshot.validateFiniteFloatingPoint()
        let initialIdentity = try weightIdentity(initialWeights(target: target, width: width, seed: seed))
        guard configuration["initial_weights_sha256"] as? String == initialIdentity else {
            throw NativeCheckpointError.invalid("compact initialization identity differs from its recorded seed")
        }
    }

    static func weightShapes(target: String, width: Int) throws -> [String: [Int]] {
        _ = try architecture(for: target)
        guard [16, 32].contains(width) else { throw StudioError("Compact material width must be 16 or 32.") }
        var shapes: [String: [Int]] = [:]
        func convolution(_ name: String, _ incoming: Int, _ outgoing: Int, kernel: Int = 1, groups: Int = 1) {
            shapes[name + ".weight"] = [outgoing, incoming / groups, kernel, kernel]
            shapes[name + ".bias"] = [1, outgoing, 1, 1]
        }
        func block(_ name: String, _ channels: Int) {
            for index in 1...2 {
                shapes[name + ".norm\(index).weight"] = [1, channels, 1, 1]
                shapes[name + ".norm\(index).bias"] = [1, channels, 1, 1]
            }
            convolution(name + ".expand", channels, channels * 2)
            convolution(name + ".depthwise", channels * 2, channels * 2, kernel: 3, groups: channels * 2)
            convolution(name + ".attention", channels, channels)
            convolution(name + ".project", channels, channels)
            convolution(name + ".ffn_expand", channels, channels * 2)
            convolution(name + ".ffn_project", channels, channels)
            shapes[name + ".beta"] = [1, channels, 1, 1]
            shapes[name + ".gamma"] = [1, channels, 1, 1]
        }
        convolution("input", 3, width, kernel: 3)
        for level in 0..<3 {
            let channels = width << level
            block("encoder.\(level)", channels)
            convolution("down.\(level)", channels, channels * 2, kernel: 2)
            convolution("up.\(level)", channels * 2, channels)
            block("decoder.\(level)", channels)
        }
        block("middle", width * 8)
        convolution("output", width, target == "normal" ? 3 : 1, kernel: 3)
        return shapes
    }

    private static func initialWeights(target: String, width: Int, seed: UInt64) throws -> [String: NativeTensor] {
        let shapes = try weightShapes(target: target, width: width)
        var random = NativeMaterialRandom(seed: seed), result: [String: NativeTensor] = [:]
        for name in shapes.keys.sorted() {
            try Task.checkCancellation()
            let shape = shapes[name]!, count = shape.reduce(1, *)
            var values: [Float]
            if name.contains(".norm"), name.hasSuffix(".weight") { values = .init(repeating: 1, count: count) }
            else if name.hasSuffix(".beta") || name.hasSuffix(".gamma") { values = .init(repeating: 0.1, count: count) }
            else if name == "output.bias" { values = target == "normal" ? [0.5, 0.5, 1] : [0.5] }
            else if name.hasSuffix(".bias") { values = .init(repeating: 0, count: count) }
            else {
                let incoming = shape[1] * shape[2] * shape[3]
                let outgoing = (name.contains("depthwise") ? 1 : shape[0]) * shape[2] * shape[3]
                let limit = sqrt(Float(6) / Float(incoming + outgoing))
                values = (0..<count).map { _ in (random.unit() * 2 - 1) * limit }
            }
            result[name] = .floats(values, shape: shape)
        }
        return result
    }

    private static func weightIdentity(_ weights: [String: NativeTensor]) throws -> String {
        let encoded = try NativeSafetensors.encoded(tensors: weights, metadata: [:])
        return SHA256.hash(data: encoded).map { String(format: "%02x", $0) }.joined()
    }

    private static func integer(_ value: Any?) -> Int? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(),
              let result = Int(number.stringValue) else { return nil }
        return result
    }
    private static func unsigned(_ value: Any?) -> UInt64? {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
        return UInt64(number.stringValue)
    }
    private static func boolean(_ value: Any?) -> Bool? {
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { return nil }
        return number.boolValue
    }
    private static func validGrid(width: Int, height: Int) -> Bool {
        width >= 64 && height >= 64 && width <= 8192 && height <= 8192 && width % 64 == 0 && height % 64 == 0
    }

    func updateWeights(_ weights: [String: NativeTensor]) throws {
        try Self.validateWeights(weights, shapes: Self.weightShapes(target: target, width: networkWidth))
        self.weights = weights
    }
    private static func validateWeights(_ weights: [String: NativeTensor], shapes: [String: [Int]]) throws {
        guard Set(weights.keys) == Set(shapes.keys) else { throw StudioError("Compact material weights are incomplete.") }
        for (name, tensor) in weights {
            guard tensor.dtype == "F32", tensor.shape == shapes[name], tensor.bytes.count == tensor.shape.reduce(4, *),
                  try tensor.floatValues().allSatisfy(\.isFinite) else {
                throw StudioError("Compact material weight shape or values are invalid: \(name).")
            }
        }
    }

    func program(width: Int, height: Int, target: String) throws -> Program {
        guard target == self.target else { throw StudioError("This compact model was trained for a different material map.") }
        if let cached, cached.width == width, cached.height == height { return cached }
        let program = try Program(model: self, width: width, height: height)
        cached = program; return program
    }
    func predict(rgb: [Float], width: Int, height: Int, target: String) throws -> NativeMaterialPrediction {
        let result = try program(width: width, height: height, target: target).execute(rgb: rgb, weights: weights)
        return NativeMaterialPrediction(width: width, height: height, channels: outputChannels, values: result.output)
    }
    func checkpointConfiguration(size: Int, step: Int, validation: [String: Any]? = nil) throws -> [String: Any] {
        guard Self.validGrid(width: size, height: size), step >= 0 else { throw StudioError("Compact checkpoint grid or step is invalid.") }
        var result = configuration
        result["schema"] = Self.schema; result["architecture"] = architectureID
        result["model_family"] = target == "normal" ? "compact-normal" : "compact-scalar"
        result["target"] = target; result["network_width"] = networkWidth
        result["input_channels"] = 3; result["output_channels"] = outputChannels
        result["from_scratch"] = true; result["initialization_seed"] = initializationSeed
        result["initial_weights_sha256"] = baseSHA256; result["training_scope"] = "all-weights"; result["scope"] = "full-model"
        result["runtime"] = "Apple MPSGraph Float32"; result["reduced_precision_fast_math"] = false
        result["head_activation"] = "linear"; result["step"] = step; result["training_size"] = size
        result["image_padding"] = false; result["image_resizing"] = false
        result["input_transfer"] = "sRGB diffuse codes / code maximum"
        result["target_transfer"] = "linear numeric source codes / code maximum"
        result["native_runtime"] = "Apple MPSGraph Float32, reduced precision fast math disabled"
        result["trained_utc"] = ISO8601DateFormatter().string(from: Date())
        result.removeValue(forKey: "base"); result.removeValue(forKey: "layers")
        if let validation { result["validation"] = validation }
        return result
    }

    final class Program {
        let width: Int, height: Int, target: String, frozenStageCount = 0
        let outputChannels: Int
        private let networkWidth: Int, shapes: [String: [Int]]
        private let graph = MPSGraph()
        private var input: MPSGraphTensor!
        private var output: MPSGraphTensor!
        private var parameters: [String: MPSGraphTensor] = [:]
        private var targetInput: MPSGraphTensor?, loss: MPSGraphTensor?, valueLoss: MPSGraphTensor?, detailLoss: MPSGraphTensor?
        private var derivatives: [String: MPSGraphTensor] = [:]
        // Validation and training have distinct executable identities. Their
        // separate caches keep a final validation from recompiling training.
        private var predictionCache: [String: MPSGraphExecutable] = [:]
        private var lossCache: [String: MPSGraphExecutable] = [:]
        private var derivativeCache: [String: MPSGraphExecutable] = [:]
        private(set) var executionStatistics = NativeGraphExecution.Statistics()

        init(model: NativeCompactMaterialModel, width: Int, height: Int) throws {
            guard NativeCompactMaterialModel.validGrid(width: width, height: height) else {
                throw StudioError("Compact material evaluation needs native dimensions divisible by 64, without resizing or padding.")
            }
            self.width = width; self.height = height; target = model.target
            networkWidth = model.networkWidth; outputChannels = model.outputChannels
            shapes = try NativeCompactMaterialModel.weightShapes(target: target, width: networkWidth)
            input = graph.placeholder(shape: [1, 3, height, width].map { NSNumber(value: $0) }, dataType: .float32, name: "compact_native_rgb")
            for name in shapes.keys.sorted() {
                parameters[name] = graph.placeholder(shape: shapes[name]!.map { NSNumber(value: $0) }, dataType: .float32, name: name)
            }
            output = try build(input)
            guard output.shape?.map(\.intValue) == [1, outputChannels, height, width] else {
                throw StudioError("Compact material graph changed the native output dimensions.")
            }
        }

        private func constant(_ value: Double) -> MPSGraphTensor { graph.constant(value, dataType: .float32) }
        private func add(_ a: MPSGraphTensor, _ b: MPSGraphTensor) -> MPSGraphTensor { graph.addition(a, b, name: nil) }
        private func multiply(_ a: MPSGraphTensor, _ b: MPSGraphTensor) -> MPSGraphTensor { graph.multiplication(a, b, name: nil) }
        private func subtract(_ a: MPSGraphTensor, _ b: MPSGraphTensor) -> MPSGraphTensor { graph.subtraction(a, b, name: nil) }
        private func slice(_ value: MPSGraphTensor, axis: Int, start: Int, length: Int) -> MPSGraphTensor {
            graph.sliceTensor(value, dimension: axis, start: start, length: length, name: nil)
        }
        private func convolution(_ input: MPSGraphTensor, _ name: String, stride: Int = 1, groups: Int = 1) throws -> MPSGraphTensor {
            guard let weight = parameters[name + ".weight"], let bias = parameters[name + ".bias"], let shape = shapes[name + ".weight"] else {
                throw StudioError("Compact material graph omitted a convolution: \(name).")
            }
            let padding = stride == 1 ? shape[2] / 2 : 0
            guard let descriptor = MPSGraphConvolution2DOpDescriptor(strideInX: stride, strideInY: stride,
                dilationRateInX: 1, dilationRateInY: 1, groups: groups,
                paddingLeft: padding, paddingRight: padding, paddingTop: padding, paddingBottom: padding,
                paddingStyle: .explicit, dataLayout: .NCHW, weightsLayout: .OIHW) else {
                throw StudioError("Compact material convolution descriptor is invalid.")
            }
            return add(graph.convolution2D(input, weights: weight, descriptor: descriptor, name: name), bias)
        }
        private func normalize(_ input: MPSGraphTensor, _ name: String) -> MPSGraphTensor {
            let mean = graph.mean(of: input, axes: [1], name: nil)
            let centered = subtract(input, mean)
            let variance = graph.mean(of: multiply(centered, centered), axes: [1], name: nil)
            let normalized = graph.division(centered, graph.squareRoot(with: add(variance, constant(1e-6)), name: nil), name: nil)
            return add(multiply(normalized, parameters[name + ".weight"]!), parameters[name + ".bias"]!)
        }
        private func gate(_ input: MPSGraphTensor, channels: Int) -> MPSGraphTensor {
            multiply(slice(input, axis: 1, start: 0, length: channels), slice(input, axis: 1, start: channels, length: channels))
        }
        private func block(_ input: MPSGraphTensor, _ name: String, channels: Int) throws -> MPSGraphTensor {
            var value = try convolution(normalize(input, name + ".norm1"), name + ".expand")
            value = try convolution(value, name + ".depthwise", groups: channels * 2)
            value = gate(value, channels: channels)
            let attention = try convolution(graph.mean(of: value, axes: [2, 3], name: nil), name + ".attention")
            value = try convolution(multiply(value, attention), name + ".project")
            let residual = add(input, multiply(value, parameters[name + ".beta"]!))
            value = try convolution(normalize(residual, name + ".norm2"), name + ".ffn_expand")
            value = try convolution(gate(value, channels: channels), name + ".ffn_project")
            return add(residual, multiply(value, parameters[name + ".gamma"]!))
        }
        private func build(_ input: MPSGraphTensor) throws -> MPSGraphTensor {
            var value = try convolution(input, "input"), skips: [MPSGraphTensor] = []
            for level in 0..<3 {
                value = try block(value, "encoder.\(level)", channels: networkWidth << level)
                skips.append(value)
                value = try convolution(value, "down.\(level)", stride: 2)
            }
            value = try block(value, "middle", channels: networkWidth * 8)
            for level in (0..<3).reversed() {
                value = try convolution(value, "up.\(level)")
                value = add(NativeGraphExecution.nearestNeighbor2(value, graph: graph), skips[level])
                value = try block(value, "decoder.\(level)", channels: networkWidth << level)
            }
            return try convolution(value, "output")
        }
        private func prepareLoss() {
            guard targetInput == nil else { return }
            let reference = graph.placeholder(shape: output.shape, dataType: .float32, name: "compact_linear_target")
            targetInput = reference
            let axes: [NSNumber] = [0, 1, 2, 3]
            let absolute = graph.mean(of: graph.absolute(with: subtract(output, reference), name: nil), axes: axes, name: nil)
            var detail = constant(0)
            for offset in [1, 2, 4, 8] {
                for axis in [2, 3] {
                    let length = (axis == 2 ? height : width) - offset
                    let predictionDelta = subtract(slice(output, axis: axis, start: offset, length: length), slice(output, axis: axis, start: 0, length: length))
                    let targetDelta = subtract(slice(reference, axis: axis, start: offset, length: length), slice(reference, axis: axis, start: 0, length: length))
                    let mean = graph.mean(of: graph.absolute(with: subtract(predictionDelta, targetDelta), name: nil), axes: axes, name: nil)
                    detail = add(detail, graph.division(mean, constant(Double(offset)), name: nil))
                }
            }
            valueLoss = absolute; detailLoss = detail; loss = add(absolute, multiply(detail, constant(4)))
        }
        private func prepareDerivatives() throws {
            guard derivatives.isEmpty else { return }
            prepareLoss()
            let names = parameters.keys.sorted()
            let gradients = graph.gradients(of: loss!, with: names.map { parameters[$0]! }, name: "compact_all_weight_gradients")
            for name in names {
                guard let derivative = gradients[parameters[name]!] else { throw StudioError("Compact material graph cannot differentiate weight: \(name).") }
                derivatives[name] = derivative
            }
        }

        func execute(rgb: [Float], weights: [String: NativeTensor], reference: [Float]? = nil,
                     gradientsOnly: Bool = false, featureKey: String? = nil,
                     checkCancellation: () throws -> Void = { try Task.checkCancellation() },
                     onStage: (Int, Int) -> Void = { _, _ in },
                     onOperation: (String, Int, Int) -> Void = { _, _, _ in }) throws -> NativeMaterialModel.Program.Execution {
            try Task.checkCancellation(); try checkCancellation()
            guard rgb.count == width * height * 3, rgb.allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }) else {
                throw StudioError("Compact material evaluation needs finite native RGB codes between zero and one.")
            }
            try NativeCompactMaterialModel.validateWeights(weights, shapes: shapes)
            if let reference {
                guard reference.count == width * height * outputChannels, reference.allSatisfy(\.isFinite) else {
                    throw StudioError("Compact material target grid or values are invalid.")
                }
                prepareLoss()
            } else if gradientsOnly { throw StudioError("Compact material gradients need a target map.") }
            if gradientsOnly { try prepareDerivatives() }
            var feeds = try Dictionary(uniqueKeysWithValues: parameters.map { name, tensor in
                (tensor, try NativeGraphExecution.tensorData(weights[name]!))
            })
            feeds[input] = try NativeGraphExecution.tensorData(.floats(rgb, shape: [1, 3, height, width]))
            var targets = [output!]
            if let reference {
                feeds[targetInput!] = try NativeGraphExecution.tensorData(.floats(reference, shape: [1, outputChannels, height, width]))
                targets += [loss!, valueLoss!, detailLoss!]
            }
            let names = gradientsOnly ? parameters.keys.sorted() : []
            targets += names.map { derivatives[$0]! }
            let label = gradientsOnly ? "Computing compact model gradients" : "Computing compact prediction"
            onOperation(label, 0, 1); onStage(0, 1)
            var cache = gradientsOnly ? derivativeCache : reference == nil ? predictionCache : lossCache
            let cold = cache.isEmpty, started = ProcessInfo.processInfo.systemUptime
            let data = try NativeGraphExecution.runData(graph, feeds: feeds, targets: targets, cache: &cache, checkCancellation: checkCancellation)
            if gradientsOnly { derivativeCache = cache } else if reference == nil { predictionCache = cache } else { lossCache = cache }
            executionStatistics.compilations += cold ? 1 : 0
            executionStatistics.stages += 1
            executionStatistics.executionSeconds += ProcessInfo.processInfo.systemUptime - started
            try Task.checkCancellation(); try checkCancellation()
            let tensors = try data.map { try NativeGraphExecution.tensor($0) }
            let prediction = try tensors[0].floatValues()
            let total = reference == nil ? nil : try tensors[1].floatValues()[0]
            let value = reference == nil ? nil : try tensors[2].floatValues()[0]
            let detail = reference == nil ? nil : try tensors[3].floatValues()[0]
            guard prediction.allSatisfy(\.isFinite), [total, value, detail].allSatisfy({ $0?.isFinite ?? true }) else {
                throw NativeMaterialSampleError(message: "Compact material model produced nonfinite sample output or loss; weights remain untouched.")
            }
            var gradients: [String: NativeTensor] = [:]
            for (index, name) in names.enumerated() {
                let gradient = tensors[index + 4]
                guard gradient.dtype == "F32", gradient.shape == shapes[name] else {
                    throw StudioError("Compact material gradient shape differs from its weight: \(name).")
                }
                guard try gradient.floatValues().allSatisfy(\.isFinite) else {
                    throw NativeMaterialSampleError(message: "Compact material gradients are nonfinite for this sample: \(name).")
                }
                gradients[name] = gradient
            }
            onOperation(label, 1, 1); onStage(1, 1)
            return .init(output: prediction, loss: total, valueLoss: value, gradientLoss: detail,
                updated: [:], optimizerState: [:], gradients: gradients)
        }
    }
}

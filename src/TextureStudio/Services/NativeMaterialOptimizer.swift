import Foundation

/// Optimizes recorded LoRA factors or complete compact model weights. Forward/backward work stays
/// in MPSGraph; keeping this update separate allows accumulation without retaining
/// a model graph or applying weights between microbatches.
struct NativeMaterialOptimizerConfiguration: Sendable, Equatable {
    var algorithm = "adamw"
    var beta1: Float = 0.9
    var beta2: Float = 0.999
    var epsilon: Float = 1e-8
    var weightDecay: Float = 0
    var maxGradientNorm: Float = 1

    var isValid: Bool {
        ["adam", "adamw"].contains(algorithm) && beta1.isFinite && beta2.isFinite &&
        (0..<1).contains(beta1) && (0..<1).contains(beta2) && epsilon.isFinite && epsilon > 0 &&
        weightDecay.isFinite && weightDecay >= 0 && maxGradientNorm.isFinite && maxGradientNorm >= 0
    }
    var metadata: [String: Any] {
        ["optimizer": algorithm, "optimizer_beta1": beta1, "optimizer_beta2": beta2,
         "optimizer_epsilon": epsilon, "weight_decay": weightDecay, "max_gradient_norm": maxGradientNorm,
         "weight_decay_mode": algorithm == "adamw" ? "decoupled" : "coupled_l2"]
    }
}

struct NativeMaterialGradientAccumulator {
    private(set) var count = 0
    private var sums: [String: [Float]] = [:]
    private var shapes: [String: [Int]] = [:]

    mutating func add(_ gradients: [String: NativeTensor]) throws {
        guard !gradients.isEmpty, count == 0 || Set(gradients.keys) == Set(sums.keys), count < Int.max else {
            throw StudioError("The accumulated adapter gradients are incomplete.")
        }
        for (name, gradient) in gradients {
            let values = try gradient.floatValues()
            guard gradient.dtype == "F32" else { throw StudioError("An adapter gradient has an unsupported type.") }
            guard values.allSatisfy(\.isFinite) else { throw NativeMaterialSampleError(message: "An adapter gradient is nonfinite for this sample.") }
            if var sum = sums[name] {
                guard shapes[name] == gradient.shape, sum.count == values.count else { throw StudioError("Accumulated adapter gradient dimensions changed.") }
                for index in sum.indices { sum[index] += values[index] }
                guard sum.allSatisfy(\.isFinite) else { throw NativeMaterialSampleError(message: "This sample overflowed accumulated adapter gradients in Float32.") }
                sums[name] = sum
            } else { sums[name] = values; shapes[name] = gradient.shape }
        }
        count += 1
    }

    func averaged() throws -> [String: NativeTensor] {
        guard count > 0 else { throw StudioError("An optimizer update needs at least one evaluated map.") }
        let divisor = Float(count)
        return Dictionary(uniqueKeysWithValues: sums.map { name, values in
            (name, NativeTensor.floats(values.map { $0 / divisor }, shape: shapes[name]!))
        })
    }
}

enum NativeMaterialOptimizer {
    struct Update {
        let weights: [String: NativeTensor]
        let state: [String: NativeTensor]
        let gradientNorm: Double
    }

    static func apply(gradients: [String: NativeTensor], weights: [String: NativeTensor],
                      state: [String: NativeTensor], learningRate: Float, step: Int,
                      configuration: NativeMaterialOptimizerConfiguration) throws -> Update {
        guard configuration.isValid, learningRate.isFinite, learningRate > 0, step > 0,
              !weights.isEmpty, Set(gradients.keys) == Set(weights.keys) else { throw StudioError("Native optimizer configuration or adapter gradients are invalid.") }
        let stateNames = Set(weights.keys.flatMap { [$0 + ".m", $0 + ".v"] })
        guard state.isEmpty || Set(state.keys) == stateNames else { throw StudioError("Native optimizer moment state is incomplete.") }
        var gradientValues: [String: [Float]] = [:]
        var normSquared = Double(0)
        for name in gradients.keys.sorted() {
            let tensor = gradients[name]!
            guard tensor.shape == weights[name]!.shape, tensor.dtype == "F32" else { throw StudioError("Native optimizer gradient dimensions differ from the adapter.") }
            let values = try tensor.floatValues()
            guard values.allSatisfy(\.isFinite) else { throw StudioError("Native optimizer received nonfinite gradients.") }
            normSquared += values.reduce(0) { $0 + Double($1) * Double($1) }
            gradientValues[name] = values
        }
        let norm = sqrt(normSquared)
        let clip = configuration.maxGradientNorm == 0 ? 1 : min(1, Double(configuration.maxGradientNorm) / (norm + 1e-6))
        let beta1 = Double(configuration.beta1), beta2 = Double(configuration.beta2)
        let correction1 = 1 - pow(beta1, Double(step)), correction2 = 1 - pow(beta2, Double(step))
        let rate = Double(learningRate), decay = Double(configuration.weightDecay), epsilon = Double(configuration.epsilon)
        var updated: [String: NativeTensor] = [:], nextState: [String: NativeTensor] = [:]
        for name in weights.keys.sorted() {
            let tensor = weights[name]!, parameters = try tensor.floatValues(), derivative = gradientValues[name]!
            guard tensor.dtype == "F32", parameters.count == derivative.count, parameters.allSatisfy(\.isFinite) else { throw StudioError("Native optimizer adapter values are invalid.") }
            func moment(_ suffix: String) throws -> [Float] {
                guard let old = state[name + suffix] else { return [Float](repeating: 0, count: parameters.count) }
                guard old.dtype == "F32", old.shape == tensor.shape else { throw StudioError("Native optimizer moment dimensions differ from the adapter.") }
                let values = try old.floatValues()
                guard values.allSatisfy(\.isFinite), suffix != ".v" || values.allSatisfy({ $0 >= 0 }) else { throw StudioError("Native optimizer moment state is invalid.") }
                return values
            }
            var m = try moment(".m"), v = try moment(".v"), values = parameters
            for index in values.indices {
                let parameter = Double(parameters[index])
                let gradient = Double(derivative[index]) * clip + (configuration.algorithm == "adam" ? decay * parameter : 0)
                m[index] = Float(beta1 * Double(m[index]) + (1 - beta1) * gradient)
                v[index] = Float(beta2 * Double(v[index]) + (1 - beta2) * gradient * gradient)
                let direction = (Double(m[index]) / correction1) / (sqrt(Double(v[index]) / correction2) + epsilon)
                let decayed = configuration.algorithm == "adamw" ? parameter * (1 - rate * decay) : parameter
                values[index] = Float(decayed - rate * direction)
            }
            guard values.allSatisfy(\.isFinite), m.allSatisfy(\.isFinite), v.allSatisfy(\.isFinite) else {
                throw NativeMaterialSampleError(message: "This sample group overflowed the optimizer; adapter weights and moments remain untouched.")
            }
            updated[name] = .floats(values, shape: tensor.shape)
            nextState[name + ".m"] = .floats(m, shape: tensor.shape)
            nextState[name + ".v"] = .floats(v, shape: tensor.shape)
        }
        return Update(weights: updated, state: nextState, gradientNorm: norm)
    }
}

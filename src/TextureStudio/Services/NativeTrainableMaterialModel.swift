import Foundation

/// Keeps the established refinement backend and compact full-weight training
/// behind the same sample-recovery, scheduling and checkpoint lifecycle.
final class NativeTrainableMaterialModel {
    private enum Backend {
        case refinement(NativeMaterialModel)
        case compact(NativeCompactMaterialModel)
    }
    private let backend: Backend

    init(_ model: NativeMaterialModel) { backend = .refinement(model) }
    init(_ model: NativeCompactMaterialModel) { backend = .compact(model) }

    static func load(_ options: NativeMaterialTrainer.Options, training: Bool) throws -> NativeTrainableMaterialModel {
        if options.modelFamily == .pbrnxt {
            return try .init(NativeMaterialModel.load(checkpointURL: options.baseline ? nil : options.checkpoint,
                expectedSHA256: options.expectedSHA256, baseURL: options.base, target: options.target,
                scope: options.scope, rank: options.rank, alpha: options.alpha, training: training, seed: options.seed))
        }
        // A comparison uses fresh weights but still requires a valid recorded
        // compact architecture and initialization identity.
        if options.baseline, let checkpoint = options.checkpoint {
            let inspected = try NativeMaterialCheckpoint.inspect(at: checkpoint, expectedSHA256: options.expectedSHA256)
            let recorded = try NativeMaterialTransfer.object(Data(inspected.utf8))
            guard recorded["schema"] as? String == NativeCompactMaterialModel.schema,
                  recorded["model_family"] as? String == options.modelFamily.rawValue,
                  let target = recorded["target"] as? String, target == options.target,
                  let width = recorded["network_width"] as? Int, width == options.compactWidth,
                  let seed = recorded["initialization_seed"] as? UInt64, seed == options.seed else {
                throw StudioError("The compact comparison must use its checkpoint's recorded family, target, width and initialization seed.")
            }
            return try .init(NativeCompactMaterialModel(target: target, width: width, seed: seed))
        }
        return try .init(NativeCompactMaterialModel.load(checkpointURL: options.baseline ? nil : options.checkpoint,
            expectedSHA256: options.expectedSHA256, target: options.target, width: options.compactWidth, seed: options.seed))
    }

    var configuration: [String: Any] {
        get {
            switch backend {
            case .refinement(let model): return model.configuration
            case .compact(let model): return model.configuration
            }
        }
        set {
            switch backend {
            case .refinement(let model): model.configuration = newValue
            case .compact(let model): model.configuration = newValue
            }
        }
    }
    var isCompact: Bool { if case .compact = backend { return true }; return false }
    var trainableWeights: [String: NativeTensor] {
        switch backend {
        case .refinement(let model): return model.adapterWeights
        case .compact(let model): return model.weights
        }
    }
    var layers: [String: NativeMaterialModel.AdapterLayer] {
        if case .refinement(let model) = backend { return model.layers }
        return [:]
    }
    var baseSHA256: String {
        switch backend {
        case .refinement(let model): return model.baseSHA256
        case .compact(let model): return model.baseSHA256
        }
    }
    var parameterCount: Int { trainableWeights.values.reduce(0) { $0 + $1.shape.reduce(1, *) } }
    func updateWeights(_ weights: [String: NativeTensor]) throws {
        switch backend {
        case .refinement(let model): model.updateAdapters(weights)
        case .compact(let model): try model.updateWeights(weights)
        }
    }
    func prepareTraining(size: Int) throws {
        if case .refinement(let model) = backend { try NativeMaterialTrainer.admitTraining(model: model, size: size) }
    }
    func predict(rgb: [Float], width: Int, height: Int, target: String) throws -> NativeMaterialPrediction {
        switch backend {
        case .refinement(let model): return try model.predict(rgb: rgb, width: width, height: height, target: target)
        case .compact(let model): return try model.predict(rgb: rgb, width: width, height: height, target: target)
        }
    }
    func checkpointConfiguration(size: Int, step: Int, validation: [String: Any]?) throws -> [String: Any] {
        switch backend {
        case .refinement(let model): return try model.checkpointConfiguration(size: size, step: step, validation: validation)
        case .compact(let model): return try model.checkpointConfiguration(size: size, step: step, validation: validation)
        }
    }
    func export(configuration: [String: Any], to output: URL, developer: Bool) throws -> String {
        switch backend {
        case .refinement(let model): return try NativeMaterialPackage.export(model: model, configuration: configuration, to: output, developer: developer)
        case .compact(let model): return try NativeMaterialPackage.export(model: model, configuration: configuration, to: output, developer: developer)
        }
    }
    func program(width: Int, height: Int, target: String) throws -> Program {
        switch backend {
        case .refinement(let model): return try .init(model.program(width: width, height: height, target: target))
        case .compact(let model): return try .init(model.program(width: width, height: height, target: target))
        }
    }

    final class Program {
        private enum Backend {
            case refinement(NativeMaterialModel.Program)
            case compact(NativeCompactMaterialModel.Program)
        }
        private let backend: Backend
        init(_ program: NativeMaterialModel.Program) { backend = .refinement(program) }
        init(_ program: NativeCompactMaterialModel.Program) { backend = .compact(program) }
        var frozenStageCount: Int {
            switch backend {
            case .refinement(let program): return program.frozenStageCount
            case .compact(let program): return program.frozenStageCount
            }
        }
        var executionStatistics: NativeGraphExecution.Statistics {
            switch backend {
            case .refinement(let program): return program.executionStatistics
            case .compact(let program): return program.executionStatistics
            }
        }
        func execute(rgb: [Float], weights: [String: NativeTensor], reference: [Float]? = nil,
                     gradientsOnly: Bool = false, featureKey: String? = nil,
                     checkCancellation: () throws -> Void = { try Task.checkCancellation() },
                     onStage: (Int, Int) -> Void = { _, _ in },
                     onOperation: (String, Int, Int) -> Void = { _, _, _ in }) throws -> NativeMaterialModel.Program.Execution {
            switch backend {
            case .refinement(let program):
                return try program.execute(rgb: rgb, adapters: weights, reference: reference, gradientsOnly: gradientsOnly,
                    featureKey: featureKey, checkCancellation: checkCancellation, onStage: onStage, onOperation: onOperation)
            case .compact(let program):
                return try program.execute(rgb: rgb, weights: weights, reference: reference, gradientsOnly: gradientsOnly,
                    featureKey: featureKey, checkCancellation: checkCancellation, onStage: onStage, onOperation: onOperation)
            }
        }
    }
}

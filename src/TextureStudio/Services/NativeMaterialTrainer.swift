import CryptoKit
import Darwin
import Foundation

final class NativeMaterialTrainingControl: @unchecked Sendable {
    private let lock = NSLock()
    private var aborted = false, finalSave = false, checkpoint = false
    func stop() { lock.withLock { aborted = true } }
    func stopAndSave() { lock.withLock { finalSave = true } }
    func saveCheckpoint() { lock.withLock { checkpoint = true } }
    var shouldStopAndSave: Bool { lock.withLock { finalSave } }
    func check() throws { try Task.checkCancellation(); if lock.withLock({ aborted }) { throw CancellationError() } }
    func consumeCheckpoint() -> Bool { lock.withLock { let requested = checkpoint; checkpoint = false; return requested } }
}

/// Native whole-grid Float32 material training. MPSGraph differentiates the
/// recorded LoRA factors or compact model weights; Swift owns averaging and Adam/AdamW updates, dataset
/// locks, iteration, validation and bit-preserving checkpoint publication.
enum NativeMaterialTrainer {
    typealias Event = @Sendable (String) -> Void
    static func run(arguments: [String], onEvent: @escaping Event = { _ in }, control: NativeMaterialTrainingControl = .init()) async throws -> String {
        try Task.checkCancellation()
        let job = Task.detached(priority: .userInitiated) {
            let options = try Options(arguments)
            if options.command == "infer" { return try infer(options, control: control) }
            return try train(options, onEvent: onEvent, control: control)
        }
        return try await withTaskCancellationHandler { try await job.value } onCancel: { control.stop(); job.cancel() }
    }
    struct Options: Sendable {
        let command: String, target: String, scope: String
        let modelFamily: MaterialTrainingModelFamily
        let compactWidth: Int
        let modelName: String?
        let dataset: URL?, output: URL, image: URL?, checkpoint: URL?, expectedSHA256: String?, base: URL
        let size: Int, rank: Int, updatesPerMap: Int, validationEvery: Int, checkpointEvery: Int
        let validationUnit: MaterialTrainingIntervalUnit, checkpointUnit: MaterialTrainingIntervalUnit
        let alpha: Float, learningRate: Float, maxMinutes: Double
        let gradientAccumulationSteps: Int, warmupUpdates: Int
        let learningRateSchedule: String, minimumLearningRateRatio: Float
        let optimizerConfiguration: NativeMaterialOptimizerConfiguration
        let seed: UInt64, materials: [String], inputEncoding: String, baseline: Bool, developerMode: Bool
        init(_ arguments: [String]) throws {
            guard let command = arguments.first, ["train", "refine", "infer"].contains(command) else { throw StudioError("Unsupported native material operation.") }
            self.command = command
            func value(_ flag: String) -> String? { guard let index = arguments.firstIndex(of: flag), arguments.indices.contains(index + 1) else { return nil }; return arguments[index + 1] }
            func url(_ flag: String) -> URL? { value(flag).map { URL(fileURLWithPath: $0).standardizedFileURL } }
            guard let output = url("--output") else { throw StudioError("Choose a new material operation output folder.") }
            self.output = output; dataset = url("--dataset"); image = url("--image"); checkpoint = url("--checkpoint")
            let requestedSHA256 = value("--expected-sha256")
            var recorded: [String: Any] = [:]
            if let checkpoint {
                let snapshot = try NativeSafetensors(contentsOf: NativeMaterialModel.resolvedCheckpoint(checkpoint), expectedSHA256: requestedSHA256)
                expectedSHA256 = snapshot.sha256
                if let text = snapshot.metadata["configuration"] { recorded = (try JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any] ?? [:] }
            } else { expectedSHA256 = requestedSHA256 }
            let requestedName = (value("--model-name") ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            let recordedName = (recorded["model_name"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            let name = requestedName.isEmpty ? recordedName : requestedName
            modelName = name.isEmpty ? nil : name
            target = value("--target") ?? recorded["target"] as? String ?? "height"
            let recordedArchitecture = recorded["architecture"] as? String
            let recordedFamily: String? = recorded["model_family"] as? String ??
                (recordedArchitecture == "texture-studio-compact-scalar-native-v1" ? "compact-scalar" :
                 recordedArchitecture == "texture-studio-compact-normal-native-v1" ? "compact-normal" : nil)
            guard let family = MaterialTrainingModelFamily(rawValue: value("--model-family") ?? recordedFamily ?? "pbrnxt") else {
                throw StudioError("Choose a supported native material model family.")
            }
            modelFamily = family
            if let recordedFamily, recordedFamily != family.rawValue {
                throw StudioError("The selected checkpoint belongs to a different material model family.")
            }
            scope = value("--scope") ?? recorded["scope"] as? String ?? (family == .pbrnxt ? "final-map" : "full-model")
            guard ["height", "roughness", "normal"].contains(target),
                  family == .pbrnxt ? ["final-map", "map-decoder"].contains(scope) : scope == "full-model",
                  family == .pbrnxt || (family.rawValue == "compact-normal" ? target == "normal" : target != "normal") else {
                throw StudioError("Choose a compatible material map and training scope for this model family.")
            }
            compactWidth = Int(value("--compact-width") ?? String(recorded["network_width"] as? Int ?? 32)) ?? 0
            guard family == .pbrnxt || [16, 32].contains(compactWidth) else { throw StudioError("Compact model width must be 16 or 32.") }
            let directory = url("--model-directory") ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!.appendingPathComponent("Texture Studio/Material Models/pbrnxt-base")
            base = url("--base-checkpoint") ?? NativeMaterialPackage.baseURL(directory, configuration: recorded)
            size = Int(value("--size") ?? (family == .pbrnxt ? "1024" : "512")) ?? 0
            rank = Int(value("--lora-rank") ?? "8") ?? 0
            alpha = Float(value("--lora-alpha") ?? "8") ?? .nan
            learningRate = Float(value("--learning-rate") ?? (family == .pbrnxt ? "0.00001" : "0.001")) ?? .nan
            gradientAccumulationSteps = Int(value("--gradient-accumulation-steps") ?? "1") ?? 0
            warmupUpdates = Int(value("--warmup-updates") ?? "0") ?? -1
            learningRateSchedule = value("--learning-rate-schedule") ?? "constant"
            let minimumRatio = Double(value("--minimum-learning-rate-ratio") ?? "0.1") ?? .nan
            minimumLearningRateRatio = minimumRatio <= 1 ? Float(minimumRatio) : .nan
            func nonnegative(_ flag: String, default fallback: Double) -> Float {
                guard let parsed = Double(value(flag) ?? String(fallback)), parsed.isFinite, parsed >= 0,
                      parsed == 0 || Float(parsed) > 0 else { return .nan }
                return Float(parsed)
            }
            optimizerConfiguration = .init(algorithm: value("--optimizer") ?? "adamw",
                beta1: nonnegative("--optimizer-beta1", default: 0.9),
                beta2: nonnegative("--optimizer-beta2", default: 0.999),
                epsilon: nonnegative("--optimizer-epsilon", default: 1e-8),
                weightDecay: nonnegative("--weight-decay", default: 0),
                maxGradientNorm: nonnegative("--max-gradient-norm", default: 1))
            updatesPerMap = Int(value("--updates-per-map") ?? "100") ?? 0
            maxMinutes = Double(value("--max-minutes") ?? "30") ?? .nan
            validationEvery = Int(value("--validation-every") ?? "0") ?? -1
            checkpointEvery = Int(value("--checkpoint-every") ?? "0") ?? -1
            guard let validationUnit = MaterialTrainingIntervalUnit(rawValue: value("--validation-unit") ?? "epoch"),
                  let checkpointUnit = MaterialTrainingIntervalUnit(rawValue: value("--checkpoint-unit") ?? "epoch") else {
                throw StudioError("Validation and checkpoint intervals must use epoch or step.")
            }
            self.validationUnit = validationUnit; self.checkpointUnit = checkpointUnit
            let recordedSeed = recorded["initialization_seed"].map { String(describing: $0) } ?? "17"
            guard let parsedSeed = UInt64(value("--seed") ?? recordedSeed) else { throw StudioError("The training seed must be a nonnegative 64-bit integer.") }
            seed = parsedSeed
            materials = arguments.indices.filter { arguments[$0] == "--material" && arguments.indices.contains($0 + 1) }.map { arguments[$0 + 1] }
            inputEncoding = value("--input-encoding") ?? "srgb"
            baseline = arguments.contains("--baseline"); developerMode = arguments.contains("--developer-mode")
            guard size >= 256, size <= 8192, size % 64 == 0,
                  family != .pbrnxt || (rank > 0 && alpha.isFinite && alpha > 0),
                  learningRate.isFinite, learningRate > 0, updatesPerMap > 0, maxMinutes.isFinite, maxMinutes > 0,
                  validationEvery >= 0, checkpointEvery >= 0 else { throw StudioError("Material training sizes, rates and update counts are invalid.") }
            guard gradientAccumulationSteps > 0, warmupUpdates >= 0, optimizerConfiguration.isValid,
                  ["constant", "cosine"].contains(learningRateSchedule), minimumLearningRateRatio.isFinite,
                  minimumLearningRateRatio > 0, minimumLearningRateRatio <= 1 else { throw StudioError("Material optimizer, gradient accumulation or learning rate schedule values are invalid.") }
            guard command == "infer" ? image != nil : dataset != nil else { throw StudioError("The native material operation needs its input image or dataset.") }
        }

        func effectiveLearningRate(update: Int, totalUpdates: Int) -> Float {
            let multiplier: Double
            if warmupUpdates > 0, update <= warmupUpdates {
                multiplier = Double(update) / Double(warmupUpdates)
            } else if learningRateSchedule == "cosine", totalUpdates > warmupUpdates, totalUpdates - warmupUpdates > 1 {
                let progress = min(1, max(0, Double(update - warmupUpdates - 1) / Double(totalUpdates - warmupUpdates - 1)))
                let floor = Double(minimumLearningRateRatio)
                multiplier = floor + (1 - floor) * (1 + cos(.pi * progress)) / 2
            } else { multiplier = 1 }
            return max(Float.leastNonzeroMagnitude, Float(Double(learningRate) * multiplier))
        }

        var trainingConfiguration: [String: Any] {
            optimizerConfiguration.metadata.merging([
                "learning_rate": learningRate, "gradient_accumulation_steps": gradientAccumulationSteps,
                "microbatch_size": 1, "effective_batch_size": gradientAccumulationSteps,
                "learning_rate_schedule": learningRateSchedule, "minimum_learning_rate_ratio": minimumLearningRateRatio,
                "minimum_learning_rate": max(Float.leastNonzeroMagnitude, learningRate * minimumLearningRateRatio),
                "warmup_updates": warmupUpdates, "seed": seed, "precision": "Float32",
                "value_loss_weight": 1, "detail_loss_weight": 4,
                "model_family": modelFamily.rawValue, "trainable_parameters": modelFamily == .pbrnxt ? "LoRA factors" : "complete compact network",
                "optimizer_state_restored": false,
                "validation_every": validationEvery, "validation_unit": validationUnit.rawValue,
                "checkpoint_every": checkpointEvery, "checkpoint_unit": checkpointUnit.rawValue,
                "sample_failure_policy": "Quarantine unreadable, changed, malformed or nonfinite samples for this run; continue with valid samples and save without requiring successful validation.",
                "gradient_accumulation_policy": "Average gradients over evaluated maps, then clip once and apply one optimizer update; stop-and-save and the deadline flush a partial group."
            ]) { _, new in new }
        }
    }
    private static func infer(_ options: Options, control: NativeMaterialTrainingControl) throws -> String {
        try control.check()
        let imageURL = options.image!, snapshot = try Data(contentsOf: imageURL)
        let photo = try NativePNG.decode(snapshot)
        let rgb = try photo.modelFloatSamples(role: "input", encoding: options.inputEncoding)
        let model = try NativeTrainableMaterialModel.load(options, training: false)
        let prediction = try model.predict(rgb: rgb, width: photo.header.width, height: photo.header.height, target: options.target)
        try control.check()
        guard try checksum(Data(contentsOf: imageURL)) == checksum(snapshot) else { throw StudioError("The prepared diffuse changed during inference.") }
        guard !FileManager.default.fileExists(atPath: options.output.path) else { throw StudioError("Choose a new inference output folder.") }
        let stage = options.output.deletingLastPathComponent().appendingPathComponent(".native-material-inference-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: false)
        defer { try? FileManager.default.removeItem(at: stage) }
        let filename = options.target + ".float32.exr", final = options.output.appendingPathComponent(filename)
        try NativeMaterialNumericExporter.writeEXR(prediction, to: stage.appendingPathComponent(filename))
        let digest = try checksum(Data(contentsOf: stage.appendingPathComponent(filename)))
        let checkpointHash = try options.checkpoint.map { checksum(try Data(contentsOf: NativeMaterialModel.resolvedCheckpoint($0))) }
        guard checkpointHash == nil || checkpointHash == options.expectedSHA256 else {
            throw StudioError("The selected checkpoint changed during inference. Choose the model again.")
        }
        let result: [String: Any] = ["checkpoint_sha256": checkpointHash ?? model.baseSHA256,
            "checkpoint_step": options.baseline ? 0 : model.configuration["step"] ?? 0,
            "target": options.target, "input_kind": "diffuse", "diffuse_path": imageURL.path,
            "model_family": options.modelFamily.rawValue, "architecture": model.configuration["architecture"] ?? "pbrnxt-native-v1",
            "output_channels": prediction.channels, "untrained_initialization": options.modelFamily != .pbrnxt && options.baseline,
            "image_sha256": checksum(snapshot), "native_dimensions": [prediction.width, prediction.height],
            "source_bits": photo.header.bits, "source_bytes_modified": false,
            "generation": ["tiled": false, "model_input_dimensions": [prediction.width, prediction.height],
                "source_pixels_resized": false, "source_pixels_discarded": false, "runtime": "Apple MPSGraph Float32"],
            "outputs": [options.target: ["path": final.path, "sha256": digest, "encoding": "linear_data", "storage": "FLOAT32", "blender_color_space": "Non-Color"]]]
        try writeJSON(result, to: stage.appendingPathComponent("inference.json"))
        try control.check()
        try FileManager.default.moveItem(at: stage, to: options.output)
        return try json(result)
    }
    static func train(_ options: Options, onEvent: Event, control: NativeMaterialTrainingControl, model suppliedModel: NativeMaterialModel? = nil,
                      compactModel suppliedCompactModel: NativeCompactMaterialModel? = nil,
                      now: @Sendable () -> ContinuousClock.Instant = { .now }) throws -> String {
        let activity = ProcessInfo.processInfo.beginActivity(options: .userInitiated, reason: "Training the requested material model")
        defer { ProcessInfo.processInfo.endActivity(activity) }
        try control.check()
        func setup(_ label: String, step: Int, requested: Int? = nil, maps: Int? = nil) throws {
            var event: [String: Any] = ["event": "training_setup", "operation": label,
                "completed": step - 1, "total": 3, "updates_per_map": options.updatesPerMap]
            if let requested { event["requested_updates"] = requested }
            if let maps { event["training_map_count"] = maps }
            onEvent(try json(event) + "\n")
        }
        try setup("Reading and verifying dataset", step: 1)
        let dataset = options.dataset!
        var directories = [dataset]
        let manifestBytes = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let manifest = try JSONSerialization.jsonObject(with: manifestBytes) as? [String: Any]
        if let lineage = manifest?["native_size_preparation"] as? [String: Any],
           let source = lineage["source_dataset_path"] as? String {
            let original = URL(fileURLWithPath: source).resolvingSymlinksInPath()
            if original != dataset.resolvingSymlinksInPath() { directories.append(original) }
        }
        let locks = try directories.sorted { $0.path < $1.path }.map(DatasetReadLock.init)
        defer { locks.forEach { $0.close() } }
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: dataset, size: options.size, target: options.target, materials: options.materials)
        let settings = manifest?["validation"] as? [String: Any] ?? [:]
        let validationEnabled = settings["enabled"] as? Bool ?? true
        let quickCount = max(0, settings["quick_count"] as? Int ?? 4)
        let training = descriptors.filter { $0.split == "train" }
        let validation = validationEnabled ? descriptors.filter { $0.split == "validation" } : []
        guard !training.isEmpty else { throw StudioError("The selected native dataset has no included training maps.") }
        let requested = training.count.multipliedReportingOverflow(by: options.updatesPerMap)
        guard !requested.overflow else { throw StudioError("The requested training update count is too large.") }
        let requestedUpdates = requested.partialValue
        guard !requestedUpdates.multipliedReportingOverflow(by: options.gradientAccumulationSteps).overflow else {
            throw StudioError("The requested accumulated map evaluation count is too large.")
        }
        let sourceHash = checksum(manifestBytes)
        guard checksum(try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))) == sourceHash else {
            throw StudioError("The prepared dataset changed before native training obtained its read locks.")
        }
        try setup("Loading material model", step: 2, requested: requestedUpdates, maps: training.count)
        let model: NativeTrainableMaterialModel
        if let suppliedModel { model = .init(suppliedModel) }
        else if let suppliedCompactModel { model = .init(suppliedCompactModel) }
        else { model = try NativeTrainableMaterialModel.load(options, training: true) }
        if let name = options.modelName { model.configuration["model_name"] = name }
        var effectiveConfiguration = options.trainingConfiguration
        effectiveConfiguration["training_size"] = options.size
        effectiveConfiguration["updates_per_map"] = options.updatesPerMap
        effectiveConfiguration["validation_every"] = options.validationEvery
        effectiveConfiguration["checkpoint_every"] = options.checkpointEvery
        effectiveConfiguration["max_minutes"] = options.maxMinutes
        effectiveConfiguration["target"] = options.target
        effectiveConfiguration["scope"] = options.scope
        effectiveConfiguration["dataset_manifest_sha256"] = sourceHash
        effectiveConfiguration["material_filter"] = options.materials
        effectiveConfiguration["requested_updates"] = requestedUpdates
        effectiveConfiguration["training_map_count"] = training.count
        effectiveConfiguration["validation_map_count"] = validation.count
        effectiveConfiguration["validation_enabled"] = validationEnabled
        effectiveConfiguration["quick_validation_count"] = quickCount
        effectiveConfiguration["initial_step"] = model.configuration["step"] as? Int ?? 0
        effectiveConfiguration["scheduling_step_origin"] = "current_run"
        if !model.isCompact { effectiveConfiguration["lora_layers"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(model.layers)) }
        effectiveConfiguration["trainable_parameter_count"] = model.parameterCount
        effectiveConfiguration["output_channels"] = options.target == "normal" ? 3 : 1
        model.configuration["training_configuration"] = effectiveConfiguration
        try model.prepareTraining(size: options.size)
        try setup("Preparing model execution graph", step: 3)
        let program = try model.program(width: options.size, height: options.size, target: options.target)
        try control.check()
        guard !FileManager.default.fileExists(atPath: options.output.path) else { throw StudioError("Choose a new training output directory.") }
        try FileManager.default.createDirectory(at: options.output, withIntermediateDirectories: true)
        let started = now()
        var completed = 0, random = NativeMaterialRandom(seed: options.seed)
        var accumulationRandom = NativeMaterialRandom(seed: options.seed ^ 0x9e3779b97f4a7c15)
        var sampleEvaluations = 0, accumulationOrder: [Int] = [], accumulationPosition = 0
        var currentEpoch = 0, activeUpdate = false, workflowPhase = 3
        var completedEpochs = 0
        var quarantined = Set<String>(), skippedSamples: [[String: Any]] = []
        let initialStep = model.configuration["step"] as? Int ?? 0
        var optimizer: [String: NativeTensor] = [:], checkpoints: [[String: Any]] = [], history: [[String: Any]] = []
        var lastSavedStep = -1, lastValidation: [String: Any] = [:]
        var ownedExport: URL?
        func seconds(_ duration: Duration) -> Double {
            let components = duration.components
            return Double(components.seconds) + Double(components.attoseconds) / 1e18
        }
        func elapsedSeconds() -> Double { seconds(started.duration(to: now())) }
        func timeLimitReached() -> Bool { elapsedSeconds() / 60 >= options.maxMinutes }
        func stoppedReason() -> String? {
            if control.shouldStopAndSave { return "user_stop" }
            if training.allSatisfy({ quarantined.contains("train:" + $0.id) }) { return "no_valid_training_samples" }
            return completed < requestedUpdates && timeLimitReached() ? "time_limit" : nil
        }
        func emit(_ event: [String: Any]) throws {
            var event = event
            event["elapsed_training_seconds"] = elapsedSeconds()
            event["completed_updates"] = completed
            event["sample_evaluations"] = sampleEvaluations
            event["skipped_sample_count"] = skippedSamples.count
            event["completed_epochs"] = completedEpochs
            event["gradient_accumulation_steps"] = options.gradientAccumulationSteps
            event["requested_updates"] = requestedUpdates
            event["current_update"] = activeUpdate ? completed + 1 : completed
            event["initial_step"] = initialStep
            event["checkpoint_step"] = initialStep + completed
            event["epoch"] = currentEpoch
            event["total_epochs"] = options.updatesPerMap
            event["workflow_phase"] = workflowPhase
            if let name = options.modelName { event["model_name"] = name }
            let line = try json(event) + "\n"; onEvent(line)
        }
        func pair(_ descriptor: NativeMaterialDatasetService.TrainingSample) throws -> ([Float], [Float]) {
            try control.check()
            let inputBytes = try Data(contentsOf: descriptor.inputURL), targetBytes = try Data(contentsOf: descriptor.targetURL)
            guard checksum(inputBytes) == descriptor.inputSHA256, checksum(targetBytes) == descriptor.targetSHA256 else {
                throw StudioError("A native training map changed after the dataset was prepared.")
            }
            let input = try NativePNG.decode(inputBytes), target = try NativePNG.decode(targetBytes)
            guard input.header.width == options.size, input.header.height == options.size,
                  target.header.width == input.header.width, target.header.height == input.header.height else { throw StudioError("Every native training pair must exactly match the prepared model grid.") }
            let rgb = try input.modelFloatSamples(role: "input", encoding: descriptor.inputEncoding)
            let reference = try target.modelFloatSamples(role: options.target, normalConvention: descriptor.targetConvention)
            return (rgb, reference)
        }
        func quarantine(_ sample: NativeMaterialDatasetService.TrainingSample, phase: String, error: Error) throws {
            // An explicit Abort or task cancellation always wins over recovery.
            if error is CancellationError { throw error }
            try control.check()
            guard quarantined.insert(sample.split + ":" + sample.id).inserted else { return }
            let issue: [String: Any] = ["sample_id": sample.id, "split": sample.split,
                "phase": phase, "error": error.localizedDescription,
                "input_path": sample.inputURL.path, "target_path": sample.targetURL.path,
                "step": initialStep + completed, "epoch": currentEpoch]
            skippedSamples.append(issue)
            try writeJSON(["samples": skippedSamples], to: options.output.appendingPathComponent("skipped-samples.json"))
            try emit(issue.merging(["event": "sample_skipped", "operation": "Skipping problematic sample"]) { _, new in new })
        }
        func availablePair(_ sample: NativeMaterialDatasetService.TrainingSample, phase: String) throws -> ([Float], [Float])? {
            try control.check()
            guard !quarantined.contains(sample.split + ":" + sample.id) else { return nil }
            do { return try pair(sample) }
            catch { try quarantine(sample, phase: phase, error: error); return nil }
        }
        func check(full: Bool, context: String) throws -> [String: Any] {
            var errors: [[String: Any]] = [], sum = Double(0)
            let available = validation.filter { !quarantined.contains($0.split + ":" + $0.id) }
            let selected = full ? available : Array(available.prefix(quickCount))
            if !selected.isEmpty {
                try emit(["event": "validation_started", "scope": full ? "full" : "quick", "context": context,
                    "sample_count": selected.count, "pool_count": validation.count])
            }
            for (position, sample) in selected.enumerated() {
                try emit(["event": "validation_sample", "scope": full ? "full" : "quick", "context": context,
                    "sample_id": sample.id, "sample_position": position + 1, "sample_total": selected.count,
                    "operation": "Loading validation maps"])
                let mae: Double? = try autoreleasepool {
                    guard let data = try availablePair(sample, phase: "validation") else { return nil }
                    do {
                        let result = try program.execute(rgb: data.0, weights: model.trainableWeights, reference: data.1,
                        featureKey: sample.inputSHA256 + ":" + sample.inputEncoding, checkCancellation: { try control.check() }, onStage: { done, total in
                            if done == 1 || done % 10 == 0 || done == total {
                                try? emit(["event": "feature_progress", "phase": "validation", "sample_id": sample.id, "completed": done, "total": total])
                            }
                        }, onOperation: { operation, done, total in
                            try? emit(["event": "operation_progress", "phase": "validation", "operation": operation,
                                "completed": done, "total": total])
                        })
                        guard let loss = result.valueLoss, loss.isFinite else {
                            throw NativeMaterialSampleError(message: "The validation sample produced a nonfinite loss.")
                        }
                        return Double(loss)
                    } catch let error as NativeMaterialSampleError {
                        try quarantine(sample, phase: "validation", error: error)
                        return nil
                    }
                }
                guard let mae else { continue }
                errors.append(["sample_id": sample.id, "mae": mae]); sum += mae
            }
            let result: [String: Any] = ["event": "validation", "status": !validationEnabled ? "disabled" : errors.isEmpty ? "unavailable" : "checked",
                "scope": full ? "full" : "quick", "context": context, "sample_count": errors.count, "pool_count": validation.count,
                "validation_skipped_sample_count": validation.filter { quarantined.contains($0.split + ":" + $0.id) }.count,
                "mae": errors.isEmpty ? NSNull() : sum / Double(errors.count), "samples": errors,
                "step": initialStep + completed, "reference": "Teacher/source agreement, not measured material accuracy"]
            history.append(result); if history.count > 200 { history.removeFirst() }
            if validationEnabled && !validation.isEmpty { try emit(result) }
            return result
        }
        func save(final: Bool = false) throws -> URL {
            if final { workflowPhase = 4 }
            if lastSavedStep == completed { return options.output.appendingPathComponent(String(format: "checkpoint-step-%08d.safetensors", initialStep + completed)) }
            lastValidation = try check(full: true, context: final ? "final" : "checkpoint")
            try control.check()
            try emit(["event": "checkpoint_started", "operation": "Writing checkpoint", "final": final])
            let config = try model.checkpointConfiguration(size: options.size, step: initialStep + completed, validation: lastValidation)
            let configText = try json(config)
            let destination = options.output.appendingPathComponent(String(format: "checkpoint-step-%08d.safetensors", initialStep + completed))
            try control.check()
            try NativeSafetensors.write(tensors: model.trainableWeights, metadata: ["configuration": configText], to: destination)
            var information = try JSONSerialization.jsonObject(with: Data(NativeMaterialCheckpoint.inspect(at: destination).utf8)) as! [String: Any]
            checkpoints.append(information)
            information["event"] = "checkpoint_saved"; information["validation"] = lastValidation
            try emit(information)
            lastSavedStep = completed
            return destination
        }
        do {
            try emit(["event": "training_started", "runtime": "Apple MPSGraph", "precision": "Float32", "training_size": options.size,
                "execution": model.isCompact ? "compact-native-full-network-v1" : "bounded-native-stages-v1", "stage_count": program.frozenStageCount,
                "model_family": options.modelFamily.rawValue, "trainable_parameter_count": model.parameterCount,
                "requested_updates": requestedUpdates, "updates_per_map": options.updatesPerMap,
                "training_map_count": training.count, "validation_map_count": validation.count,
                "training_configuration": effectiveConfiguration,
                "quick_count": quickCount, "validation_every": options.validationEvery, "max_minutes": options.maxMinutes])
            let baseline = try check(full: true, context: "baseline")
            for epoch in 0..<options.updatesPerMap {
                currentEpoch = epoch + 1
                var order = Array(training.indices); random.shuffle(&order)
                var epochFinished = true
                for (position, index) in order.enumerated() {
                    try control.check()
                    // A deadline schedules no new update. A step already in
                    // progress finishes before the validated final save.
                    if control.shouldStopAndSave || timeLimitReached() { epochFinished = false; break }
                    let sample = training[index]
                    if quarantined.contains(sample.split + ":" + sample.id) { continue }
                    activeUpdate = true
                    try emit(["event": "update_started", "sample_id": sample.id, "sample_position": position + 1,
                        "sample_total": training.count, "operation": "Loading training maps"])
                    let updateStarted = now()
                    var accumulated = NativeMaterialGradientAccumulator()
                    var evaluatedSamples: [NativeMaterialDatasetService.TrainingSample] = []
                    var valueLoss = Double(0), detailLoss = Double(0), totalLoss = Double(0)
                    var dataPreparationSeconds = 0.0
                    let rate = options.effectiveLearningRate(update: completed + 1, totalUpdates: requestedUpdates)
                    for microbatch in 0..<options.gradientAccumulationSteps {
                        let accumulatedSample: NativeMaterialDatasetService.TrainingSample
                        if microbatch == 0 { accumulatedSample = sample }
                        else {
                            if accumulationPosition == accumulationOrder.count {
                                accumulationOrder = Array(training.indices); accumulationRandom.shuffle(&accumulationOrder)
                                accumulationPosition = 0
                            }
                            accumulatedSample = training[accumulationOrder[accumulationPosition]]
                            accumulationPosition += 1
                        }
                        try emit(["event": "accumulation_sample", "sample_id": accumulatedSample.id,
                            "accumulation_step": microbatch + 1, "learning_rate": rate,
                            "operation": "Accumulating map gradients"])
                        let derivative: NativeMaterialModel.Program.Execution? = try autoreleasepool {
                            let loadingStarted = ProcessInfo.processInfo.systemUptime
                            guard let data = try availablePair(accumulatedSample, phase: "training") else { return nil }
                            dataPreparationSeconds += ProcessInfo.processInfo.systemUptime - loadingStarted
                            do { return try program.execute(rgb: data.0, weights: model.trainableWeights, reference: data.1,
                            gradientsOnly: true,
                            featureKey: accumulatedSample.inputSHA256 + ":" + accumulatedSample.inputEncoding, checkCancellation: { try control.check() }, onStage: { done, total in
                                if done == 1 || done % 10 == 0 || done == total {
                                    try? emit(["event": "feature_progress", "phase": "training", "sample_id": accumulatedSample.id, "completed": done, "total": total,
                                        "accumulation_step": microbatch + 1])
                                }
                            }, onOperation: { operation, done, total in
                                try? emit(["event": "operation_progress", "phase": "training", "operation": operation,
                                    "completed": done, "total": total, "accumulation_step": microbatch + 1])
                            })
                            } catch let error as NativeMaterialSampleError {
                                try quarantine(accumulatedSample, phase: "training", error: error)
                                return nil
                            }
                        }
                        if let derivative {
                            // Validate before committing any compact sums. A bad
                            // microbatch cannot contaminate its valid neighbours.
                            guard let value = derivative.valueLoss, let detail = derivative.gradientLoss,
                                  let loss = derivative.loss, value.isFinite, detail.isFinite, loss.isFinite else {
                                try quarantine(accumulatedSample, phase: "training", error: StudioError("The training sample produced nonfinite losses."))
                                if control.shouldStopAndSave || timeLimitReached() { break }
                                continue
                            }
                            var candidate = accumulated
                            do { try candidate.add(derivative.gradients) }
                            catch let error as NativeMaterialSampleError { try quarantine(accumulatedSample, phase: "training", error: error)
                                if control.shouldStopAndSave || timeLimitReached() { break }
                                continue
                            }
                            accumulated = candidate
                            evaluatedSamples.append(accumulatedSample)
                            valueLoss += Double(value); detailLoss += Double(detail); totalLoss += Double(loss)
                            sampleEvaluations += 1
                        }
                        try control.check()
                        if control.shouldStopAndSave || timeLimitReached() { break }
                    }
                    try control.check()
                    guard accumulated.count > 0 else {
                        activeUpdate = false
                        if control.consumeCheckpoint() { _ = try save() }
                        if control.shouldStopAndSave || timeLimitReached() { epochFinished = false; break }
                        continue
                    }
                    try emit(["event": "operation_progress", "phase": "training", "operation": "Applying optimizer update", "completed": 0, "total": 1])
                    let optimizerStarted = ProcessInfo.processInfo.systemUptime
                    let update: NativeMaterialOptimizer.Update
                    do {
                        update = try NativeMaterialOptimizer.apply(gradients: accumulated.averaged(), weights: model.trainableWeights,
                            state: optimizer, learningRate: rate, step: completed + 1, configuration: options.optimizerConfiguration)
                    } catch let error as NativeMaterialSampleError {
                        for sample in evaluatedSamples { try quarantine(sample, phase: "training", error: error) }
                        activeUpdate = false
                        if control.consumeCheckpoint() { _ = try save() }
                        if control.shouldStopAndSave || timeLimitReached() { epochFinished = false; break }
                        continue
                    }
                    let optimizerSeconds = ProcessInfo.processInfo.systemUptime - optimizerStarted
                    try control.check()
                    try model.updateWeights(update.weights); optimizer = update.state
                    try emit(["event": "operation_progress", "phase": "training", "operation": "Applying optimizer update", "completed": 1, "total": 1])
                    completed += 1
                    activeUpdate = false
                    try emit(["event": "update", "step": initialStep + completed, "sample_id": evaluatedSamples[0].id,
                        "sample_ids": evaluatedSamples.map(\.id),
                        "value_l1": valueLoss / Double(accumulated.count), "detail_l1": detailLoss / Double(accumulated.count), "total": totalLoss / Double(accumulated.count),
                        "learning_rate": rate, "accumulated_samples": accumulated.count, "gradient_norm": update.gradientNorm,
                        "native_dimensions": [options.size, options.size],
                        "data_preparation_seconds": dataPreparationSeconds, "optimizer_seconds": optimizerSeconds,
                        "cumulative_engine_statistics": try JSONSerialization.jsonObject(with: JSONEncoder().encode(program.executionStatistics)),
                        "update_duration_seconds": seconds(updateStarted.duration(to: now()))])
                    if control.shouldStopAndSave || timeLimitReached() { epochFinished = position == order.count - 1; break }
                    // A full checkpoint validation at this epoch boundary also
                    // satisfies a quick check due at the same weight snapshot.
                    let epochCheckpointDue = position == order.count - 1 && cadenceDue(every: options.checkpointEvery,
                        unit: options.checkpointUnit, steps: completed, epochs: completedEpochs + 1, epochBoundary: true)
                    if control.consumeCheckpoint() || cadenceDue(every: options.checkpointEvery, unit: options.checkpointUnit,
                        steps: completed, epochs: completedEpochs, epochBoundary: false) { _ = try save() }
                    else if !epochCheckpointDue, quickCount > 0, !validation.isEmpty, cadenceDue(every: options.validationEvery, unit: options.validationUnit,
                        steps: completed, epochs: completedEpochs, epochBoundary: false) { _ = try check(full: false, context: "periodic") }
                    if control.shouldStopAndSave || timeLimitReached() { epochFinished = position == order.count - 1; break }
                }
                if epochFinished {
                    completedEpochs += 1
                    try emit(["event": "epoch_completed"])
                    if !control.shouldStopAndSave && !timeLimitReached() {
                        if cadenceDue(every: options.checkpointEvery, unit: options.checkpointUnit,
                            steps: completed, epochs: completedEpochs, epochBoundary: true) { _ = try save() }
                        else if lastSavedStep != completed, quickCount > 0, !validation.isEmpty, cadenceDue(every: options.validationEvery, unit: options.validationUnit,
                            steps: completed, epochs: completedEpochs, epochBoundary: true) { _ = try check(full: false, context: "periodic") }
                    }
                }
                if control.shouldStopAndSave || timeLimitReached() { break }
                if training.allSatisfy({ quarantined.contains($0.split + ":" + $0.id) }) { break }
            }
            try control.check()
            if let reason = stoppedReason() {
                try emit(["event": "training_stopped", "stopped_reason": reason,
                    "completed_updates": completed, "requested_updates": requestedUpdates, "max_minutes": options.maxMinutes])
            }
            let checkpoint = try save(final: true)
            try control.check()
            try emit(["event": "export_started", "operation": "Exporting material model", "developer_mode": options.developerMode])
            let export = options.output.appendingPathComponent("export")
            let configuration = try model.checkpointConfiguration(size: options.size, step: initialStep + completed, validation: lastValidation)
            guard !FileManager.default.fileExists(atPath: export.path) else { throw StudioError("Choose a new model export directory.") }
            // The run owns its newly created output directory. Track this new
            // child before export: cancellation can occur after publication
            // while the exporter inspects its completed package.
            ownedExport = export
            _ = try model.export(configuration: configuration, to: export, developer: options.developerMode)
            try control.check()
            let reason = stoppedReason()
            var result: [String: Any] = ["status": reason == nil ? "completed" : "stopped",
                "checkpoint_path": export.appendingPathComponent(model.isCompact || options.developerMode ? "model.safetensors" : "adapter.safetensors").path,
                "model_family": options.modelFamily.rawValue, "output_channels": options.target == "normal" ? 3 : 1,
                "package_path": export.path, "completed_updates": completed, "training_performed": completed > 0,
                "requested_updates": requestedUpdates, "updates_per_map": options.updatesPerMap,
                "training_configuration": effectiveConfiguration, "sample_evaluations": sampleEvaluations,
                "completed_epochs": completedEpochs, "skipped_sample_count": skippedSamples.count, "skipped_samples": skippedSamples,
                "max_minutes": options.maxMinutes, "elapsed_training_seconds": elapsedSeconds(),
                "time_limit_policy": "Do not start an update after the deadline. The clock starts after setup; baseline validation, an active update, and final validation and saving finish before returning.",
                "baseline_validation": baseline, "final_validation": lastValidation, "validation_history": history,
                "checkpoints": checkpoints, "last_checkpoint": checkpoint.path,
                "dataset_manifest_sha256": sourceHash, "native_dimensions": [options.size, options.size],
                "image_padding": false, "image_resizing": false, "runtime": "Apple MPSGraph Float32"]
            if let reason { result["stopped_reason"] = reason }
            if let name = options.modelName { result["model_name"] = name }
            try writeJSON(result, to: options.output.appendingPathComponent("run.json"))
            try emit(["event": "training_completed", "status": reason == nil ? "completed" : "stopped"])
            return try json(result)
        } catch {
            if error is CancellationError, let ownedExport { try? FileManager.default.removeItem(at: ownedExport) }
            var failure: [String: Any] = ["status": error is CancellationError ? "aborted" : "failed", "completed_updates": completed,
                "requested_updates": requestedUpdates, "max_minutes": options.maxMinutes,
                "training_configuration": effectiveConfiguration, "sample_evaluations": sampleEvaluations,
                "completed_epochs": completedEpochs, "skipped_sample_count": skippedSamples.count, "skipped_samples": skippedSamples,
                "elapsed_training_seconds": elapsedSeconds(), "error": error.localizedDescription,
                "training_performed": completed > 0]
            if let name = options.modelName { failure["model_name"] = name }
            try? writeJSON(failure, to: options.output.appendingPathComponent("run.json"))
            throw error
        }
    }
    static func cadenceDue(every: Int, unit: MaterialTrainingIntervalUnit, steps: Int, epochs: Int, epochBoundary: Bool) -> Bool {
        guard every > 0 else { return false }
        if unit == .epoch { return epochBoundary && epochs > 0 && epochs % every == 0 }
        return !epochBoundary && steps > 0 && steps % every == 0
    }
    // Admission bounds one checkpointed reverse stage and the largest forward
    // live set. Every recorded adapter participates; total network depth does
    // not multiply the active graph allocation.
    static func estimatedWorkingBytes(model: NativeMaterialModel, size: Int) -> UInt64 {
        let a = model.architecture
        let branch = ["normal": 1, "roughness": 2, "height": 3][model.configuration["target"] as? String ?? "height"] ?? 3
        let focused = model.layers.keys.allSatisfy { $0 == "ups.\(branch).model.10" }
        let stageChannels = max(a.dim * 24, (a.rrdbWidth + a.growth * 4) * 8)
        let activeChannels = focused ? max(stageChannels, 128) : max(stageChannels, (a.rrdbWidth + a.growth * 4) * 12)
        return UInt64(size) * UInt64(size) * UInt64(activeChannels) * 4 + UInt64(3 * 1_073_741_824)
    }
    static func admitTraining(model: NativeMaterialModel, size: Int, budget: UInt64 = MachineResources.current.maximumTrainingBytes) throws {
        let estimate = estimatedWorkingBytes(model: model, size: size)
        guard estimate <= budget else {
            throw StudioError("This checkpoint's trained layers need about \(String(format: "%.1f", Double(estimate) / 1_073_741_824)) GiB at \(size) × \(size), above this Mac's safe training budget. Close other memory-heavy work or choose a smaller grid.")
        }
    }
    static func checksum(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    static func json(_ object: [String: Any]) throws -> String { String(decoding: try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys]), as: UTF8.self) }
    private static func writeJSON(_ object: [String: Any], to url: URL) throws { try Data(json(object).utf8).write(to: url, options: .atomic) }
    private final class DatasetReadLock {
        var descriptor: Int32
        init(_ directory: URL) throws {
            descriptor = Darwin.open(directory.appendingPathComponent(".material-workbench.lock").path, O_CREAT | O_RDWR | O_NOFOLLOW, S_IRUSR | S_IWUSR)
            guard descriptor >= 0 else { throw StudioError("The training dataset cannot be locked.") }
            guard flock(descriptor, LOCK_SH | LOCK_NB) == 0 else { Darwin.close(descriptor); descriptor = -1; throw StudioError("The dataset is being modified in another window.") }
        }
        func close() { if descriptor >= 0 { flock(descriptor, LOCK_UN); Darwin.close(descriptor); descriptor = -1 } }
        deinit { close() }
    }
}

/// Direct standard FLOAT32 EXR scanlines avoid Core Image's HALF export and
/// any color-management/renderer conversion on a numeric model prediction.
enum NativeMaterialNumericExporter {
    static func writeEXR(_ prediction: NativeMaterialPrediction, to url: URL) throws {
        let width = prediction.width, height = prediction.height, channelCount = prediction.channels
        guard [1, 3].contains(channelCount), width > 0, height > 0,
              prediction.values.count == width * height * channelCount, prediction.values.allSatisfy(\.isFinite) else { throw StudioError("Invalid native material FLOAT32 image.") }
        let channels = channelCount == 1 ? ["R"] : ["B", "G", "R"]
        func u32(_ value: UInt32) -> Data { var value = value.littleEndian; return withUnsafeBytes(of: &value) { Data($0) } }
        func u64(_ value: UInt64) -> Data { var value = value.littleEndian; return withUnsafeBytes(of: &value) { Data($0) } }
        func text(_ value: String) -> Data { Data(value.utf8) + Data([0]) }
        func attribute(_ name: String, _ type: String, _ value: Data) -> Data { text(name) + text(type) + u32(UInt32(value.count)) + value }
        var header = u32(20_000_630) + u32(2), list = Data()
        for channel in channels { list.append(text(channel) + u32(2) + Data([0, 0, 0, 0]) + u32(1) + u32(1)) }
        list.append(0)
        header.append(attribute("channels", "chlist", list)); header.append(attribute("compression", "compression", Data([0])))
        let window = u32(0) + u32(0) + u32(UInt32(width - 1)) + u32(UInt32(height - 1))
        header.append(attribute("dataWindow", "box2i", window)); header.append(attribute("displayWindow", "box2i", window))
        header.append(attribute("lineOrder", "lineOrder", Data([0])))
        header.append(attribute("pixelAspectRatio", "float", u32(Float(1).bitPattern)))
        header.append(attribute("screenWindowCenter", "v2f", u32(0) + u32(0)))
        header.append(attribute("screenWindowWidth", "float", u32(Float(1).bitPattern))); header.append(0)
        let rowBytes = width * channelCount * 4, first = header.count + height * 8
        for row in 0..<height { header.append(u64(UInt64(first + row * (rowBytes + 8)))) }
        guard FileManager.default.createFile(atPath: url.path, contents: nil) else { throw StudioError("The native EXR output could not be created.") }
        let handle = try FileHandle(forWritingTo: url); defer { try? handle.close() }
        try handle.write(contentsOf: header)
        for row in 0..<height {
            try Task.checkCancellation()
            var chunk = u32(UInt32(row)) + u32(UInt32(rowBytes))
            for channel in channels {
                let component = channel == "B" ? 2 : channel == "G" ? 1 : 0
                let start = component * width * height + row * width
                prediction.values[start..<start + width].withUnsafeBytes { chunk.append(contentsOf: $0) }
            }
            try handle.write(contentsOf: chunk)
        }
        try handle.synchronize()
    }
}

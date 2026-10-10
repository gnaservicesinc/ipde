import Foundation

/// The run's optimizer position is independent of whichever setup, validation,
/// checkpoint or numeric operation is currently executing.
struct WorkbenchTrainingProgress: Equatable, Sendable {
    enum State: String, Sendable {
        case preparing, modelSetup, training, validation, checkpoint, export
        case completed, stopped, aborted, failed
    }

    static let phaseLabels = ["Preparation", "Model setup", "Training", "Saving"]
    var state: State = .preparing
    var phaseIndex = 1
    var phaseCount: Int { Self.phaseLabels.count }
    var phaseLabel: String { Self.phaseLabels[phaseIndex - 1] }
    var operationLabel = "Preparing training dataset"
    var operationDetail = ""
    var completedUpdates = 0
    var totalUpdates = 0
    var currentUpdate = 0
    var initialStep = 0
    var checkpointStep = 0
    var epoch = 0
    var totalEpochs = 0
    var samplePosition = 0
    var sampleTotal = 0
    var sampleID: String?
    var stageCompleted = 0
    var stageTotal = 0
    var elapsedSeconds = 0.0
    var lastUpdateSeconds: Double?
    var learningRate: Double?
    var accumulationStep = 0
    var gradientAccumulationSteps = 1
    var lastLoss: Double?
    var validationError: Double?

    var fractionCompleted: Double {
        totalUpdates > 0 ? Double(completedUpdates) / Double(totalUpdates) : 0
    }
    var updateSummary: String { "Total steps: \(completedUpdates.formatted()) / \(totalUpdates.formatted())" }
    var currentUpdateSummary: String? {
        guard currentUpdate > completedUpdates else { return nil }
        return "Running step \(currentUpdate.formatted()) of \(totalUpdates.formatted())"
    }
    var epochSummary: String? {
        guard epoch > 0, totalEpochs > 0 else { return nil }
        return "Epoch \(epoch.formatted()) / \(totalEpochs.formatted())"
    }
    var sampleSummary: String? {
        guard samplePosition > 0, sampleTotal > 0 else { return nil }
        let name = state == .validation ? "Validation crop" : "Training map"
        return "\(name) \(samplePosition.formatted()) / \(sampleTotal.formatted())"
    }
    var stageSummary: String? {
        guard stageTotal > 0 else { return nil }
        return "\(stageCompleted.formatted()) / \(stageTotal.formatted()) stages complete"
    }

    mutating func consume(_ event: [String: Any]) {
        guard let kind = event["event"] as? String else { return }
        if let value = event["completed_updates"] as? Int, value >= 0 { completedUpdates = value }
        if let value = event["requested_updates"] as? Int, value >= 0 { totalUpdates = value }
        if let value = event["current_update"] as? Int, value >= 0 { currentUpdate = value }
        if let value = event["initial_step"] as? Int, value >= 0 { initialStep = value }
        if let value = event["checkpoint_step"] as? Int, value >= 0 { checkpointStep = value }
        if let value = event["epoch"] as? Int, value >= 0 { epoch = value }
        if let value = event["total_epochs"] as? Int ?? event["updates_per_map"] as? Int, value >= 0 { totalEpochs = value }
        if let value = event["elapsed_training_seconds"] as? Double, value.isFinite, value >= 0 { elapsedSeconds = value }
        if let value = event["learning_rate"] as? Double, value.isFinite, value >= 0 { learningRate = value }
        if let value = event["gradient_accumulation_steps"] as? Int, value > 0 { gradientAccumulationSteps = value }
        if let value = event["accumulation_step"] as? Int, value >= 0 { accumulationStep = value }
        if let value = event["workflow_phase"] as? Int, (1...phaseCount).contains(value) { phaseIndex = value }

        switch kind {
        case "preparation_started", "preparation_progress", "preparation_completed":
            state = .preparing; phaseIndex = 1
            operationLabel = "Preparing native training maps"
            if let completed = event["completed"] as? Int, let total = event["total"] as? Int, completed >= 0, completed <= total {
                stageCompleted = completed; stageTotal = total
                operationDetail = "\(completed.formatted()) / \(total.formatted()) materials prepared"
            }
        case "source_recovery_progress":
            operationLabel = event["message"] as? String ?? "Verifying original source maps"
        case "training_setup":
            state = .modelSetup; phaseIndex = 2; operationDetail = ""
            operationLabel = event["operation"] as? String ?? "Preparing model"
            setStage(event)
        case "training_started":
            state = .training; phaseIndex = 3
            operationLabel = "Starting training"; operationDetail = ""; clearStageAndSample()
            totalEpochs = event["updates_per_map"] as? Int ?? totalEpochs
        case "update_started":
            state = .training; phaseIndex = 3
            operationLabel = event["operation"] as? String ?? "Starting optimizer update"
            operationDetail = ""; stageCompleted = 0; stageTotal = 0
            setSample(event)
        case "accumulation_sample":
            state = .training
            operationLabel = event["operation"] as? String ?? "Accumulating gradients"
            setSample(event)
        case "operation_progress":
            operationLabel = event["operation"] as? String ?? operationLabel
            if event["phase"] as? String == "training" { state = .training }
            else if event["phase"] as? String == "validation" { state = .validation }
            setStage(event)
        case "feature_progress":
            // Older runtimes supply feature counts without the labeled numeric
            // operations. They remain intelligible when replaying their logs.
            if event["workflow_phase"] == nil {
                operationLabel = "Forward features"
                setStage(event)
            }
        case "update":
            if event["completed_updates"] == nil, let step = event["step"] as? Int {
                completedUpdates = max(0, step - initialStep); currentUpdate = completedUpdates
                checkpointStep = step
            }
            state = .training
            operationLabel = "Optimizer update completed"; stageCompleted = 0; stageTotal = 0
            lastLoss = event["total"] as? Double
            if let seconds = event["update_duration_seconds"] as? Double, seconds.isFinite, seconds >= 0 {
                lastUpdateSeconds = seconds
            }
            accumulationStep = 0
        case "validation_started", "validation_sample":
            state = .validation
            switch event["context"] as? String {
            case "baseline": operationDetail = "Baseline validation before step 1"
            case "final": operationDetail = "Final validation before saving"
            case "checkpoint": operationDetail = "Full validation before checkpoint"
            default: operationDetail = event["scope"] as? String == "quick" ? "Periodic quick check" : "Full validation"
            }
            operationLabel = kind == "validation_sample" ? "Loading validation maps" : operationDetail
            stageCompleted = 0; stageTotal = 0
            if kind == "validation_sample" { setSample(event) }
            else { samplePosition = 0; sampleTotal = event["sample_count"] as? Int ?? 0; sampleID = nil }
        case "validation":
            validationError = event["mae"] as? Double
            operationLabel = "Validation completed"; clearStageAndSample()
        case "checkpoint_started":
            state = .checkpoint
            operationLabel = "Writing checkpoint"; operationDetail = ""; clearStageAndSample()
        case "checkpoint_saved":
            state = .checkpoint
            operationLabel = "Checkpoint saved"; operationDetail = ""; clearStageAndSample()
        case "training_stopped":
            operationDetail = event["stopped_reason"] as? String == "time_limit" ? "Time limit reached; saving completed steps" : "Stopping and saving completed steps"
        case "export_started":
            state = .export; phaseIndex = 4
            operationLabel = "Exporting material model"; operationDetail = ""; clearStageAndSample()
        case "training_completed":
            finish(state: event["status"] as? String == "stopped" ? .stopped : .completed)
        default: break
        }
    }

    mutating func finish(state finalState: State, completed: Int? = nil, total: Int? = nil) {
        if let completed { completedUpdates = completed }
        currentUpdate = completedUpdates
        if let total { totalUpdates = total }
        checkpointStep = initialStep + completedUpdates
        state = finalState
        if finalState == .completed || finalState == .stopped { phaseIndex = 4 }
        switch finalState {
        case .completed: operationLabel = "Training complete · model saved"
        case .stopped: operationLabel = "Training stopped · model saved"
        case .aborted: operationLabel = "Training aborted"
        case .failed: operationLabel = "Training failed"
        default: break
        }
        operationDetail = ""; clearStageAndSample()
    }

    private mutating func setSample(_ event: [String: Any]) {
        samplePosition = event["sample_position"] as? Int ?? 0
        sampleTotal = event["sample_total"] as? Int ?? 0
        sampleID = event["sample_id"] as? String
    }
    private mutating func setStage(_ event: [String: Any]) {
        guard let completed = event["completed"] as? Int, let total = event["total"] as? Int,
              total > 0, completed >= 0, completed <= total else { return }
        stageCompleted = completed; stageTotal = total
    }
    private mutating func clearStageAndSample() {
        stageCompleted = 0; stageTotal = 0
        samplePosition = 0; sampleTotal = 0; sampleID = nil
    }
}

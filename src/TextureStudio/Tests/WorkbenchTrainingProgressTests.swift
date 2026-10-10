import XCTest
@testable import TextureStudio

final class WorkbenchTrainingProgressTests: XCTestCase {
    func testUpdateTimingAndAccumulationAreIndependentOfOptimizerStepCount() {
        var progress = WorkbenchTrainingProgress()
        progress.consume(["event": "training_started", "gradient_accumulation_steps": 4, "learning_rate": 0.00001])
        progress.consume(["event": "update_started", "completed_updates": 2, "current_update": 3])
        progress.consume(["event": "accumulation_sample", "accumulation_step": 2, "gradient_accumulation_steps": 4, "sample_id": "stone"])
        XCTAssertEqual(progress.completedUpdates, 2)
        XCTAssertEqual(progress.currentUpdate, 3)
        XCTAssertEqual(progress.accumulationStep, 2)
        XCTAssertEqual(progress.gradientAccumulationSteps, 4)
        XCTAssertEqual(progress.sampleID, "stone")
        progress.consume(["event": "update", "completed_updates": 3, "current_update": 3,
            "learning_rate": 0.000002, "update_duration_seconds": 75.4])
        XCTAssertEqual(progress.completedUpdates, 3)
        XCTAssertEqual(progress.accumulationStep, 0)
        XCTAssertEqual(progress.lastUpdateSeconds, 75.4)
        XCTAssertEqual(progress.learningRate, 0.000002)
        progress.consume(["event": "update", "update_duration_seconds": Double.nan, "learning_rate": Double.infinity])
        XCTAssertEqual(progress.lastUpdateSeconds, 75.4)
        XCTAssertEqual(progress.learningRate, 0.000002)
    }

    func testSetupTrainingValidationAndSavingKeepSeparateCounters() throws {
        var progress = WorkbenchTrainingProgress()
        progress.consume(["event": "preparation_progress", "completed": 2, "total": 6])
        XCTAssertEqual(progress.phaseIndex, 1)
        XCTAssertEqual(progress.stageCompleted, 2)
        progress.consume(["event": "training_setup", "operation": "Loading material model", "completed": 1, "total": 3,
            "requested_updates": 12, "updates_per_map": 4])
        XCTAssertEqual(progress.phaseIndex, 2)
        XCTAssertEqual(progress.totalUpdates, 12)
        XCTAssertEqual(progress.totalEpochs, 4)
        progress.consume(["event": "training_started", "requested_updates": 12, "updates_per_map": 4, "initial_step": 100])
        XCTAssertNil(progress.currentUpdateSummary)
        XCTAssertNil(progress.epochSummary)
        progress.consume(["event": "update_started", "completed_updates": 4, "current_update": 5, "checkpoint_step": 104,
            "epoch": 2, "sample_position": 2, "sample_total": 3, "sample_id": "soil"])
        XCTAssertEqual(progress.currentUpdateSummary, "Running step 5 of 12")
        XCTAssertEqual(progress.epochSummary, "Epoch 2 / 4")
        progress.consume(["event": "operation_progress", "phase": "training", "operation": "Backward pass", "completed": 3, "total": 8])
        XCTAssertEqual(progress.operationLabel, "Backward pass")
        XCTAssertEqual(progress.sampleID, "soil")
        XCTAssertEqual(progress.stageCompleted, 3)
        XCTAssertEqual(progress.stageTotal, 8)
        progress.consume(["event": "update", "completed_updates": 5, "current_update": 5, "checkpoint_step": 105, "total": 0.2])
        progress.consume(["event": "validation_started", "context": "checkpoint", "scope": "full", "sample_count": 2])
        XCTAssertNil(progress.sampleID)
        XCTAssertEqual(progress.samplePosition, 0)
        progress.consume(["event": "validation_sample", "context": "checkpoint", "sample_position": 2, "sample_total": 2, "sample_id": "rock"])
        progress.consume(["event": "operation_progress", "phase": "validation", "operation": "Forward pass", "completed": 7, "total": 10])
        XCTAssertEqual(progress.state, .validation)
        XCTAssertEqual(progress.completedUpdates, 5)
        XCTAssertEqual(progress.totalUpdates, 12)
        XCTAssertEqual(progress.epoch, 2)
        XCTAssertEqual(progress.sampleID, "rock")
        XCTAssertEqual(progress.sampleSummary, "Validation crop 2 / 2")
        XCTAssertEqual(progress.phaseIndex, 3)
        progress.consume(["event": "validation_started", "context": "final", "sample_count": 2, "workflow_phase": 4])
        XCTAssertEqual(progress.phaseIndex, 4)
        XCTAssertNil(progress.sampleID)
        progress.consume(["event": "checkpoint_started"])
        XCTAssertEqual(progress.state, .checkpoint)
        XCTAssertEqual(progress.stageTotal, 0)
        progress.consume(["event": "export_started", "elapsed_training_seconds": 34.5])
        XCTAssertEqual(progress.state, .export)
        XCTAssertEqual(progress.completedUpdates, 5)
        XCTAssertNil(progress.sampleSummary)
        XCTAssertEqual(progress.elapsedSeconds, 34.5)
        progress.consume(["event": "training_completed", "status": "stopped"])
        XCTAssertEqual(progress.state, .stopped)
        XCTAssertEqual(progress.checkpointStep, 105)
        XCTAssertNil(progress.currentUpdateSummary)
        XCTAssertEqual(progress.fractionCompleted, 5.0 / 12.0, accuracy: 0.00001)
    }

    func testAbortAndFailureDoNotClaimAnUnfinishedStepIsStillRunning() {
        for finalState in [WorkbenchTrainingProgress.State.aborted, .failed] {
            var progress = WorkbenchTrainingProgress()
            progress.consume(["event": "update_started", "completed_updates": 1, "current_update": 2, "requested_updates": 5])
            XCTAssertNotNil(progress.currentUpdateSummary)
            progress.finish(state: finalState)
            XCTAssertEqual(progress.completedUpdates, 1)
            XCTAssertNil(progress.currentUpdateSummary)
        }
    }

    @MainActor func testStoreRetainsProgressDuringGracefulStopAndClearsItForAnotherOperation() async throws {
        let suite = "training-progress-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults)
        store.operation("Training", training: true) {
            store.recordTrainingProgress("{\"event\":\"training_started\",\"requested_updates\":5,\"updates_per_map\":5}\n")
            store.recordTrainingProgress("{\"event\":\"update_started\",\"completed_updates\":1,\"current_update\":2}\n")
            store.stopAndSave()
            store.recordTrainingProgress("{\"event\":\"operation_progress\",\"phase\":\"training\",\"operation\":\"Backward pass\",\"completed\":3,\"total\":8}\n")
            XCTAssertEqual(store.trainingProgress?.operationLabel, "Backward pass")
            XCTAssertEqual(store.trainingProgress?.stageCompleted, 3)
            store.recordTrainingProgress("{\"event\":\"validation\",\"sample_count\":0,\"pool_count\":0,\"mae\":null}\n")
            XCTAssertEqual(store.validationSummary, "")
            store.recordTrainingProgress("{\"event\":\"training_completed\",\"status\":\"stopped\",\"completed_updates\":2,\"current_update\":2}\n")
        }
        while store.isBusy { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertEqual(store.trainingProgress?.state, .stopped)
        XCTAssertEqual(store.trainingProgress?.completedUpdates, 2)
        store.operation("Open another dataset") { }
        XCTAssertNil(store.trainingProgress)
        while store.isBusy { try await Task.sleep(for: .milliseconds(5)) }
    }
}

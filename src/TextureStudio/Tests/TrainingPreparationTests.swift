import AppKit
import Foundation
import SwiftUI
import XCTest
@testable import TextureStudio

@MainActor
final class TrainingPreparationTests: XCTestCase {
    func testPreparationUsesSupportedNativeCropGrid() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store()
        store.preparationWorkers = 1
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        XCTAssertEqual(store.supportedTrainingSizes, [512, 1024])
        store.selectTrainingSize(1024)
        try await settled(store)
        XCTAssertFalse(fixture.calls.contains { $0.first == "prepare-size" }, "Choosing a preview grid cannot allocate a training dataset")
        store.prepareTrainingDataset()
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertTrue(store.dataset?.hasNativeSize(1024) == true)
        XCTAssertEqual(fixture.calls.last?.first, "prepare-size")
        XCTAssertEqual(value("--size", in: fixture.calls.last!), "1024")
        XCTAssertEqual(value("--preparation-workers", in: fixture.calls.last!), "1")
        XCTAssertEqual(value("--expected-index-sha256", in: fixture.calls.last!), "source-sha")
        XCTAssertEqual(try Data(contentsOf: fixture.original.appendingPathComponent("dataset.json")), Data("original".utf8))
        let count = fixture.calls.count
        store.selectTrainingSize(2048)
        XCTAssertNotNil(store.error)
        XCTAssertEqual(fixture.calls.count, count, "An unsupported size never allocates training images")
    }

    func testSelectedCenterCropNeedsNoHeldOutMaterialToBeReady() throws {
        let document = """
        {"dataset_path":"/stage","index_sha256":"bound","materials":[{"material_id":"soil_4k",
        "samples":[{"sample_id":"soil_4k_center","status":"approved","split":"train","width":2048,"height":2048,
        "maps":{"input":{"path":"/crop/diffuse.png","width":2048,"height":2048},
        "height":{"path":"/crop/height.png","width":2048,"height":2048}}}]}],
        "automatic_validation":{"policy":"subject-extra-crops-v2","material_ids":[],
        "quick_fit_material_id":"soil_4k","target":"height"}}
        """.replacingOccurrences(of: "\n", with: "")
        let dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: document)
        XCTAssertTrue(dataset.readyForTraining(size: 2048, material: "soil_4k", target: "height"))
        XCTAssertFalse(dataset.readyForTraining(size: 2048, material: nil, target: "height"))
        XCTAssertFalse(dataset.readyForTraining(size: 2048, material: "soil_4k", target: "normal"), "A height-only preparation cannot train a normal model")
        XCTAssertFalse(dataset.readyForTraining(size: 2048, material: "other", target: "height"))
    }

    func testLegacyPreparationWithUnusedMapsMustBeRegeneratedForSelectedTarget() throws {
        let document = """
        {"dataset_path":"/stage","index_sha256":"bound","materials":[{"material_id":"soil",
        "samples":[{"sample_id":"soil-center","status":"approved","split":"train","width":256,"height":256,
        "maps":{"input":{"path":"/crop/diffuse.png","width":256,"height":256},
        "height":{"path":"/crop/height.png","width":256,"height":256},
        "normal":{"path":"/crop/normal.png","width":256,"height":256}}}]}],
        "automatic_validation":{"policy":"subject-extra-crops-v2","material_ids":[],"quick_fit_material_id":"soil"}}
        """
        let dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: document)
        XCTAssertFalse(dataset.readyForTraining(size: 256, material: "soil", target: "height"))
        XCTAssertFalse(dataset.readyForTraining(size: 256, material: "soil", target: "normal"))
    }

    func testMismatchedMapGridCannotBecomeTrainingInput() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.wrongGrid = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.selectTrainingSize(1024)
        try await settled(store)
        store.prepareTrainingDataset()
        try await settled(store)
        XCTAssertNotNil(store.error)
        XCTAssertEqual(store.dataset?.datasetPath, fixture.original.path)
        XCTAssertFalse(fixture.calls.contains { $0.first == "train" })
    }

    func testTrainingDispatchesExactGridThenPurgesStagingAndReconnectsSources() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let oldDeveloperMode = StudioPreferences.defaults.object(forKey: StudioPreferences.developerModeKey)
        StudioPreferences.defaults.set(false, forKey: StudioPreferences.developerModeKey)
        defer { StudioPreferences.defaults.set(oldDeveloperMode, forKey: StudioPreferences.developerModeKey) }
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.training.modelName = "  石 Stone / Displacement  "
        store.training.learningRate = 0.0002
        store.training.gradientAccumulationSteps = 4
        store.training.optimizer = "adamw"
        store.training.optimizerBeta1 = 0.85
        store.training.optimizerBeta2 = 0.98
        store.training.optimizerEpsilon = 0.0000001
        store.training.weightDecay = 0.01
        store.training.maxGradientNorm = 0.5
        store.training.learningRateSchedule = "cosine"
        store.training.minimumLearningRateRatio = 0.2
        store.training.warmupUpdates = 10
        store.training.seed = 42
        store.training.validationEvery = 2
        store.training.validationUnit = .epoch
        store.training.checkpointEvery = 3
        store.training.checkpointUnit = .step
        store.startTraining()
        try await settled(store)
        XCTAssertNil(store.error)
        let train = try XCTUnwrap(fixture.calls.first { $0.first == "train" })
        let preparation = try XCTUnwrap(fixture.calls.first { $0.first == "prepare-size" })
        XCTAssertEqual(value("--preparation-workers", in: preparation), String(store.preparationWorkers))
        XCTAssertTrue(train.contains("--whole-maps"))
        XCTAssertEqual(value("--size", in: train), "1024")
        XCTAssertEqual(value("--dataset", in: train), fixture.prepared.path)
        XCTAssertEqual(value("--model-name", in: train), "石 Stone / Displacement")
        XCTAssertEqual(value("--learning-rate", in: train), String(store.training.learningRate))
        XCTAssertEqual(value("--gradient-accumulation-steps", in: train), "4")
        XCTAssertEqual(value("--optimizer", in: train), "adamw")
        XCTAssertEqual(value("--optimizer-beta1", in: train), "0.85")
        XCTAssertEqual(value("--optimizer-beta2", in: train), "0.98")
        XCTAssertEqual(value("--optimizer-epsilon", in: train), String(store.training.optimizerEpsilon))
        XCTAssertEqual(value("--weight-decay", in: train), "0.01")
        XCTAssertEqual(value("--max-gradient-norm", in: train), "0.5")
        XCTAssertEqual(value("--learning-rate-schedule", in: train), "cosine")
        XCTAssertEqual(value("--minimum-learning-rate-ratio", in: train), "0.2")
        XCTAssertEqual(value("--warmup-updates", in: train), "10")
        XCTAssertEqual(value("--seed", in: train), "42")
        XCTAssertEqual(value("--validation-every", in: train), "2")
        XCTAssertEqual(value("--validation-unit", in: train), "epoch")
        XCTAssertEqual(value("--checkpoint-every", in: train), "3")
        XCTAssertEqual(value("--checkpoint-unit", in: train), "step")
        XCTAssertTrue(value("--output", in: train)?.hasPrefix(fixture.root.path + "/out/material-training/material-height-") == true,
            "Display names never become path components")
        XCTAssertFalse(train.contains("--developer-mode"))
        XCTAssertTrue(fixture.calls.contains { $0.first == "cleanup-size" })
        XCTAssertEqual(store.dataset?.datasetPath, fixture.original.path)
        XCTAssertEqual(store.selectedCheckpoint?.url.pathExtension, "safetensors")
        XCTAssertTrue(store.selectedCheckpoint?.supportsTrainingWarmStart == true)
    }

    func testBlankModelNameUsesSourceDatasetAndMapBeforePreparingTrainingFiles() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store()
        store.uploadAfterTraining = false
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.training.modelName = " \n "
        let suggested = "\(store.datasetName) Displacement"
        XCTAssertEqual(store.suggestedTrainingModelName, suggested)
        store.startTraining()
        try await settled(store)
        XCTAssertNil(store.error)
        let train = try XCTUnwrap(fixture.calls.first { $0.first == "train" })
        XCTAssertEqual(value("--model-name", in: train), suggested)
        XCTAssertEqual(value("--validation-every", in: train), "1")
        XCTAssertEqual(value("--validation-unit", in: train), "epoch")
        XCTAssertEqual(value("--checkpoint-unit", in: train), "epoch")
    }

    func testStopFinalizesOnlyAfterTrainingStartsAndAllowsFinalAdapterResult() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.holdTraining = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.startTraining()
        while fixture.continuation == nil { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(store.canStopAndSave)
        store.recordTrainingProgress("{\"event\":\"training_")
        XCTAssertFalse(store.canStopAndSave, "Partial log chunks cannot enable saving")
        store.recordTrainingProgress("started\"}\n")
        XCTAssertTrue(store.canStopAndSave)
        store.stop()
        XCTAssertTrue(store.isStopping)
        XCTAssertTrue(store.isSavingTraining)
        XCTAssertTrue(store.canAbort, "Abort must remain available while Stop saves")
        store.stop()
        XCTAssertTrue(store.isSavingTraining, "A repeated Stop must not turn into Abort")
        fixture.continuation?.resume()
        fixture.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertNotNil(store.selectedCheckpoint)
        XCTAssertTrue(fixture.calls.contains { $0.first == "cleanup-size" })
    }

    func testTimeLimitedTrainingReportsPartialSavedCountsAndCleansPreparation() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.trainingResult = ["status": "stopped", "stopped_reason": "time_limit",
            "completed_updates": 1, "requested_updates": 200]
        fixture.holdTraining = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.training.scope = "map-decoder"
        store.startTraining()
        while fixture.continuation == nil { try await Task.sleep(for: .milliseconds(5)) }
        store.recordTrainingProgress("{\"event\":\"training_started\"}\n")
        store.recordTrainingProgress("{\"event\":\"training_stopped\",\"stopped_reason\":\"time_limit\",\"completed_updates\":1,\"requested_updates\":200}\n")
        XCTAssertFalse(store.canStopAndSave)
        XCTAssertTrue(store.isSavingTraining)
        XCTAssertTrue(store.canAbort, "Abort remains available during automatic final saving")
        fixture.continuation?.resume(); fixture.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.activity, "Training time limit reached. Saved 1 of 200 updates.")
        XCTAssertEqual(store.training.scope, "map-decoder")
        XCTAssertNotNil(store.selectedCheckpoint)
        XCTAssertTrue(fixture.calls.contains { $0.first == "cleanup-size" })
        XCTAssertEqual(store.dataset?.datasetPath, fixture.original.path)
    }

    func testCheckpointRequestKeepsTrainingActiveAndRegistersSavedModel() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.holdTraining = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.uploadAfterTraining = false
        store.training.size = 1024
        store.startTraining()
        for _ in 0..<200 {
            if fixture.continuation != nil { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTAssertNotNil(fixture.continuation)
        store.recordTrainingProgress("{\"event\":\"training_started\"}\n")
        store.saveCheckpointNow()
        XCTAssertTrue(store.isCheckpointPending)
        XCTAssertFalse(store.isStopping)
        store.recordTrainingProgress("{\"event\":\"validation\",\"scope\":\"full\",\"sample_count\":5,\"pool_count\":5,\"mae\":0.025}\n")
        let event: [String: Any] = ["event": "checkpoint_saved", "checkpoint_path": "/tmp/step-2.safetensors",
            "sha256": "checkpoint-proof", "schema": "texture-studio-material-lora-v1", "target": "height",
            "step": 2, "compatible": true, "variant": "lora", "supports_training_warm_start": true]
        store.recordTrainingProgress(String(decoding: try JSONSerialization.data(withJSONObject: event), as: UTF8.self) + "\n")
        XCTAssertFalse(store.isCheckpointPending)
        XCTAssertTrue(store.isTraining)
        XCTAssertTrue(store.validationSummary.contains("5/5"))
        XCTAssertEqual(store.checkpoints.last?.step, 2)
        fixture.continuation?.resume(); fixture.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
    }

    func testSkippedSamplesAndUnavailableValidationKeepRunActive() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.holdTraining = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.startTraining()
        for _ in 0..<200 {
            if fixture.continuation != nil { break }
            try await Task.sleep(for: .milliseconds(5))
        }
        store.recordTrainingProgress("{\"event\":\"training_started\"}\n")
        store.recordTrainingProgress("{\"event\":\"sample_skipped\",\"sample_id\":\"bad-map\",\"phase\":\"training\",\"error\":\"Unreadable PNG\",\"skipped_sample_count\":1}\n")
        XCTAssertTrue(store.isTraining)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.activity, "Skipped bad-map: Unreadable PNG · Training continues…")
        store.recordTrainingProgress("{\"event\":\"validation\",\"scope\":\"full\",\"status\":\"unavailable\",\"sample_count\":0,\"pool_count\":2,\"mae\":null,\"skipped_sample_count\":3,\"validation_skipped_sample_count\":2}\n")
        XCTAssertEqual(store.validationSummary, "Full validation unavailable: 0/2 crops · 2 skipped. Training and saving continue.")
        store.recordTrainingProgress("{\"event\":\"validation\",\"scope\":\"quick\",\"sample_count\":1,\"pool_count\":2,\"mae\":0.025,\"validation_skipped_sample_count\":1}\n")
        XCTAssertTrue(store.validationSummary.contains("1/2 crops"))
        XCTAssertTrue(store.validationSummary.contains("1 skipped"))
        XCTAssertTrue(store.isTraining)
        XCTAssertNil(store.error)
        fixture.continuation?.resume(); fixture.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
    }

    func testAbortDuringDatasetPreparationPreventsTrainingFromLaunching() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.holdPreparation = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.startTraining()
        while fixture.continuation == nil { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(store.canStopAndSave)
        store.abort()
        fixture.continuation?.resume(); fixture.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertFalse(fixture.calls.contains { $0.first == "train" })
        XCTAssertNil(store.selectedCheckpoint)
        XCTAssertEqual(try Data(contentsOf: fixture.original.appendingPathComponent("dataset.json")), Data("original".utf8))
    }

    func testPreparationReportsParallelProgressAndIgnoresEventsAfterStopping() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.holdPreparation = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.prepareTrainingDataset()
        while fixture.continuation == nil { try await Task.sleep(for: .milliseconds(5)) }
        let initialActivity = store.activity
        store.recordTrainingProgress("{\"event\":\"preparation_progress\",\"completed\":2,")
        XCTAssertEqual(store.activity, initialActivity)
        store.recordTrainingProgress("\"total\":8,\"worker_count\":4,\"training_size\":1024}\n")
        XCTAssertTrue(store.activity.contains("2/8 materials"))
        XCTAssertTrue(store.activity.contains("4 workers"))
        XCTAssertFalse(store.hasTrainingStarted)
        store.stop()
        let stoppedActivity = store.activity
        store.recordTrainingProgress("{\"event\":\"preparation_completed\",\"completed\":8,\"total\":8,\"worker_count\":4,\"training_size\":1024}\n")
        XCTAssertEqual(store.activity, stoppedActivity)
        fixture.continuation?.resume(); fixture.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.dataset?.datasetPath, fixture.original.path)
    }

    func testSaveRequestDuringModelSetupAbortsWithoutLoadingAnAdapter() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.holdTraining = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.startTraining()
        while fixture.continuation == nil { try await Task.sleep(for: .milliseconds(5)) }
        store.stopAndSave()
        XCTAssertTrue(store.isStopping)
        XCTAssertFalse(store.isSavingTraining)
        fixture.continuation?.resume(); fixture.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertNil(store.selectedCheckpoint)
        XCTAssertFalse(fixture.calls.contains { $0.first == "checkpoint" })
        XCTAssertTrue(fixture.calls.contains { $0.first == "cleanup-size" })
        XCTAssertEqual(store.dataset?.datasetPath, fixture.original.path)
    }

    func testAbortCancelsPendingStopWithoutLoadingOrUploadingAnAdapter() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        fixture.holdTraining = true
        let store = fixture.store()
        try await store.loadTrainingCapabilities()
        try await store.loadDataset(fixture.original)
        store.training.size = 1024
        store.startTraining()
        while fixture.continuation == nil { try await Task.sleep(for: .milliseconds(5)) }
        store.recordTrainingProgress("{\"event\":\"training_started\"}\n")
        store.stop()
        XCTAssertTrue(store.isSavingTraining)
        store.abort()
        XCTAssertFalse(store.isSavingTraining)
        XCTAssertFalse(store.canAbort)
        store.recordTrainingProgress("{\"event\":\"training_started\"}\n")
        XCTAssertFalse(store.canStopAndSave, "Delayed worker logs cannot reenable a cancelled run")
        fixture.continuation?.resume(); fixture.continuation = nil
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertNil(store.selectedCheckpoint)
        XCTAssertFalse(fixture.calls.contains { $0.first == "checkpoint" || $0.first == "upload-selected" })
        XCTAssertTrue(fixture.calls.contains { $0.first == "cleanup-size" })
    }

    /// A remote accessibility client drives these real buttons. Querying the
    /// SwiftUI accessibility tree inside its app host does not initialize it.
    /// This verifies UI routing and store transitions; trainer tests separately
    /// verify numeric updates, safetensors publication, and cancellation.
    func testExternallyDrivenTrainingStopAbortButtons() async throws {
        guard ProcessInfo.processInfo.environment["TEXTURE_STUDIO_CONTROL_UI_VERIFY"] == "1" else {
            throw XCTSkip("Set TEXTURE_STUDIO_CONTROL_UI_VERIFY=1 and use a remote UI client to press the training controls.")
        }
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store()
        store.training.scope = "map-decoder"
        let scope = store.training.scope
        let original = try Data(contentsOf: fixture.original.appendingPathComponent("dataset.json"))
        let stateURL = URL(fileURLWithPath: "/tmp/ipde-control-ui-state.json")
        var proofs: [String] = []
        func publish(_ phase: String, status: String = "waiting") throws {
            let state: [String: Any] = ["status": status, "phase": phase, "scope": store.training.scope,
                "is_busy": store.isBusy, "is_training": store.isTraining, "training_started": store.hasTrainingStarted,
                "is_stopping": store.isStopping, "is_saving": store.isSavingTraining,
                "can_stop": store.canStopAndSave, "can_abort": store.canAbort,
                "saved_checkpoint_count": store.checkpoints.count, "proofs": proofs,
                "measurement_scope": "Real SwiftUI buttons and store lifecycle; the update and checkpoint event are controlled test gates."]
            let data = try JSONSerialization.data(withJSONObject: state, options: [.sortedKeys, .prettyPrinted])
            try data.write(to: stateURL, options: .atomic)
            print("TRAINING_CONTROL_UI_STATE " + String(decoding: data, as: UTF8.self))
        }
        func until(_ label: String, _ condition: () -> Bool) async throws {
            let deadline = ContinuousClock.now.advanced(by: .seconds(90))
            while !condition() {
                guard ContinuousClock.now < deadline else {
                    store.abort()
                    try publish(label, status: "timed_out")
                    throw StudioError("The remote UI client did not complete \(label) within 90 seconds.")
                }
                try await Task.sleep(for: .milliseconds(25))
            }
        }
        let controller = NSHostingController(rootView: controlVerificationView(store, phase: "Setup: press Abort. Stop is disabled."))
        let window = NSWindow(contentRect: CGRect(x: 120, y: 120, width: 680, height: 190),
                              styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.title = "Training Stop and Abort Verification"
        window.contentViewController = controller
        defer { store.abort(); window.close() }
        NSApp.activate()
        window.makeKeyAndOrderFront(nil)

        store.operation("Training setup verification…", training: true) {
            while true { try await Task.sleep(for: .milliseconds(25)) }
        }
        try publish("setup_abort")
        try await until("setup_abort") { !store.isBusy }
        XCTAssertNil(store.error)
        XCTAssertTrue(store.activity.contains("aborted"))
        XCTAssertTrue(store.checkpoints.isEmpty)
        XCTAssertEqual(store.training.scope, scope)
        proofs.append("setup_abort")

        controller.rootView = controlVerificationView(store, phase: "Active update: press Stop to finish and save.")
        store.operation("Training Stop verification…", training: true) {
            store.recordTrainingProgress("{\"event\":\"training_started\"}\n")
            while !store.isStopping { try await Task.sleep(for: .milliseconds(25)) }
            guard store.isSavingTraining else { throw StudioError("Press Stop for the save scenario.") }
            // Hold the current update long enough to observe the saving state.
            try publish("stop_finishing_update")
            try await Task.sleep(for: .milliseconds(250))
            let checkpoint: [String: Any] = ["event": "checkpoint_saved", "checkpoint_path": fixture.root.appendingPathComponent("ui-gate.safetensors").path,
                "sha256": String(repeating: "a", count: 64), "schema": "texture-studio-material-lora-v1",
                "target": "height", "scope": scope, "step": 1, "compatible": true, "variant": "lora", "supports_training_warm_start": true]
            store.recordTrainingProgress(try fixture.json(checkpoint) + "\n")
        }
        try await until("stop_ready") { store.canStopAndSave }
        try publish("active_stop")
        try await until("active_stop") { !store.isBusy }
        XCTAssertNil(store.error)
        XCTAssertEqual(store.checkpoints.count, 1)
        XCTAssertEqual(store.checkpoints.first?.step, 1)
        XCTAssertEqual(store.training.scope, scope)
        proofs.append("stop_registered_checkpoint")

        controller.rootView = controlVerificationView(store, phase: "Press Stop, then press Abort while saving remains pending.")
        var observedPendingSave = false
        store.operation("Training pending-save Abort verification…", training: true) {
            store.recordTrainingProgress("{\"event\":\"training_started\"}\n")
            while !store.isStopping { try await Task.sleep(for: .milliseconds(25)) }
            guard store.isSavingTraining else { throw StudioError("Press Stop before aborting the pending save.") }
            observedPendingSave = true
            try publish("pending_save_abort")
            while true { try await Task.sleep(for: .milliseconds(25)) }
        }
        try await until("pending_save_ready") { store.canStopAndSave }
        try publish("active_stop_then_abort")
        try await until("pending_save_abort") { !store.isBusy }
        XCTAssertTrue(observedPendingSave)
        XCTAssertNil(store.error)
        XCTAssertTrue(store.activity.contains("aborted"))
        XCTAssertEqual(store.checkpoints.count, 1, "Abort must retain the previous saved checkpoint without registering a new one.")
        XCTAssertEqual(store.training.scope, scope)
        XCTAssertEqual(WorkbenchPreferences.load(from: fixture.defaults).training?.scope, scope)
        XCTAssertEqual(try Data(contentsOf: fixture.original.appendingPathComponent("dataset.json")), original)
        proofs.append("abort_pending_save_kept_previous_checkpoint")
        try publish("complete", status: "passed")
    }

    private func controlVerificationView(_ store: WorkbenchStore, phase: String) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(phase).font(.headline)
            Text("Refinement scope: \(store.training.scope)")
            HStack { WorkbenchStopButtons(store: store) }
            Text(store.activity).font(.caption)
        }.padding(20).frame(width: 680, height: 190, alignment: .leading)
    }

    func testMissingOnlyRemovalNeverRemovesExistingSource() async throws {
        let fixture = try Fixture()
        defer { fixture.remove() }
        let store = fixture.store()
        try await store.loadDataset(fixture.original)
        let source = fixture.root.appendingPathComponent("source.png")
        try Data("original image".utf8).write(to: source)
        let count = fixture.calls.count
        store.removeMissingSource(source, sampleID: "soil")
        XCTAssertEqual(fixture.calls.count, count)
        try FileManager.default.removeItem(at: source)
        store.removeMissingSource(source, sampleID: "soil")
        try await settled(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(fixture.calls.suffix(2).compactMap(\.first), ["remove-missing", "dataset"])
    }

    private func value(_ key: String, in args: [String]) -> String? {
        guard let i = args.firstIndex(of: key), args.indices.contains(i + 1) else { return nil }
        return args[i + 1]
    }
    private func settled(_ store: WorkbenchStore) async throws {
        for _ in 0..<1000 {
            if !store.isBusy { return }
            try await Task.sleep(for: .milliseconds(5))
        }
        XCTFail("Worker did not finish")
    }
    @MainActor private final class Fixture {
        let root: URL
        let original: URL
        let prepared: URL
        let defaults: UserDefaults
        let suite = "training-grid-\(UUID().uuidString)"
        var calls: [[String]] = []
        var wrongGrid = false
        var holdTraining = false
        var holdPreparation = false
        var trainingResult: [String: Any] = [:]
        var continuation: CheckedContinuation<Void, Never>?
        init() throws {
            root = FileManager.default.temporaryDirectory.appendingPathComponent("training-grid-\(UUID().uuidString)")
            original = root.appendingPathComponent("original")
            prepared = root.appendingPathComponent(".training-data/grid-1024")
            defaults = UserDefaults(suiteName: suite)!
            try FileManager.default.createDirectory(at: original, withIntermediateDirectories: true)
            try Data("original".utf8).write(to: original.appendingPathComponent("dataset.json"))
            defaults.set(root.path, forKey: "workspace")
        }
        func remove() { defaults.removePersistentDomain(forName: suite); try? FileManager.default.removeItem(at: root) }
        func json(_ value: [String: Any]) throws -> String {
            String(decoding: try JSONSerialization.data(withJSONObject: value), as: UTF8.self)
        }
        func dataset(prepared isPrepared: Bool) throws -> String {
            let size = isPrepared ? 1024 : 2048
            let maps: [String: Any] = ["input": ["path": root.appendingPathComponent("diffuse.png").path, "width": size, "height": size],
                "height": ["path": root.appendingPathComponent("height.png").path, "width": wrongGrid && isPrepared ? 512 : size, "height": size]]
            var value: [String: Any] = ["dataset_path": isPrepared ? prepared.path : original.path,
                "index_sha256": isPrepared ? "prepared-sha" : "source-sha", "supported_training_sizes": [512, 1024, 2048],
                "automatic_validation": ["policy": "subject-extra-crops-v2", "material_ids": ["soil"]],
                "materials": [["material_id": "soil", "samples": [["sample_id": "soil", "status": "approved", "split": "train", "width": size, "height": size, "maps": maps]]]]]
            if isPrepared { value["preparation"] = ["source_dataset_path": original.path, "source_index_sha256": "source-sha",
                "prepared_dataset_path": prepared.path, "crop_size": size, "reused": false, "target_resized": false, "target_cropped": true,
                "original_dataset_modified": false] }
            return try json(value)
        }
        func store() -> WorkbenchStore {
            WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { args, script in
                self.calls.append(args)
                switch args.first {
                case "capabilities":
                    XCTAssertEqual(script, args.first)
                    return "{\"training_sizes\":[512,1024]}"
                case "dataset", "edit-dataset": return try self.dataset(prepared: false)
                case "prepare-size":
                    if self.holdPreparation { await withCheckedContinuation { self.continuation = $0 } }
                    return try self.dataset(prepared: true)
                case "train":
                    if self.holdTraining { await withCheckedContinuation { self.continuation = $0 } }
                    var result = self.trainingResult
                    result["checkpoint_path"] = self.root.appendingPathComponent("adapter.safetensors").path
                    result["package_path"] = self.root.path
                    return try self.json(result)
                case "checkpoint": return try self.json(["checkpoint_path": self.root.appendingPathComponent("adapter.safetensors").path,
                    "sha256": "exact", "schema": "texture-studio-material-lora-v1", "target": "height", "step": 3,
                    "compatible": true, "variant": "lora", "supports_training_warm_start": true])
                case "cleanup-size": return try self.json(["dataset_path": self.prepared.path, "source_dataset_path": self.original.path, "removed": true])
                case "remove-missing": return "{\"removed\":true}"
                default: throw StudioError("Unexpected worker command")
                }
            })
        }
    }
}

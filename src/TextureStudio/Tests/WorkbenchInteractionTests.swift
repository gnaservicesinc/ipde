import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class WorkbenchInteractionTests: XCTestCase {
    func testCompactRefinementAdoptsFamilyAndClearsWarmStartWhenTargetOrFamilyChanges() throws {
        let defaults = try isolatedPreferences()
        defer { defaults.removePersistentDomain(forName: defaultsSuite) }
        let store = WorkbenchStore(preferences: defaults)
        let model = try WorkbenchResult.decode(WorkbenchCheckpoint.self,
            output: compactCheckpointJSON(id: "height-model", target: "height", family: .compactScalar))
        store.checkpoints = [model]
        XCTAssertTrue(store.selectCheckpointForRefinement(model))
        XCTAssertEqual(store.training.modelFamily, .compactScalar)
        XCTAssertEqual(store.training.effectiveScope, "full-model")
        XCTAssertEqual(store.training.learningRate, 0.001)
        XCTAssertTrue(store.training.useWarmStart)
        XCTAssertEqual(store.selectedCheckpointId, model.id)
        store.training.learningRate = 0.0007
        store.selectTrainingTarget("roughness")
        XCTAssertEqual(store.training.modelFamily, .compactScalar)
        XCTAssertEqual(store.training.learningRate, 0.0007)
        XCTAssertNil(store.selectedCheckpointId, "A height checkpoint cannot warm start a roughness model")
        XCTAssertFalse(store.training.useWarmStart)
        XCTAssertTrue(store.selectCheckpointForRefinement(model))
        store.selectTrainingModelFamily(.pbrnxt)
        XCTAssertEqual(store.training.scope, "final-map")
        XCTAssertNil(store.selectedCheckpointId)
        XCTAssertFalse(store.training.useWarmStart)
        var options = MaterialTrainingOptions()
        options.modelFamily = .compactNormal
        options.target = "normal"
        let mismatched = try WorkbenchResult.decode(WorkbenchCheckpoint.self,
            output: compactCheckpointJSON(id: "wrong-contract", target: "height", family: .compactNormal))
        XCTAssertFalse(mismatched.supportsTrainingWarmStart)
        XCTAssertFalse(mismatched.supportsStudioInference)
        XCTAssertFalse(mismatched.matchesTraining(options))
    }

    func testCompactPackageExportIgnoresHiddenPBRAggregationAndPreservesLegacyExport() throws {
        let defaults = try isolatedPreferences()
        defer { defaults.removePersistentDomain(forName: defaultsSuite) }
        let store = WorkbenchStore(preferences: defaults)
        store.modelDirectory = "/missing-pbr-base"
        store.adapterMix = [WorkbenchAdapterWeight(path: "/models/legacy-adapter.safetensors")]
        let model = try WorkbenchResult.decode(WorkbenchCheckpoint.self,
            output: compactCheckpointJSON(id: "compact", target: "normal", family: .compactNormal))
        let compact = store.checkpointPackageArguments(for: model, output: URL(fileURLWithPath: "/exports/compact"), developer: true)
        XCTAssertEqual(compact, ["package", "--checkpoint", model.checkpointPath, "--expected-sha256", model.sha256,
                                 "--output", "/exports/compact", "--developer-mode"])
        let pbr = try checkpoint(id: "legacy", target: "normal", scope: "map-decoder")
        let legacy = store.checkpointPackageArguments(for: pbr, output: URL(fileURLWithPath: "/exports/pbr"), developer: false)
        XCTAssertTrue(legacy.contains("--model-directory"))
        XCTAssertTrue(legacy.contains("--adapter"))
        XCTAssertTrue(legacy.contains("/models/legacy-adapter.safetensors=1.0"))
        XCTAssertFalse(legacy.contains("--developer-mode"))
    }

    func testCompactTrainerHandoffRestoresExactFamilyAndTypedTrainingControls() async throws {
        let defaults = try isolatedPreferences()
        defer { defaults.removePersistentDomain(forName: defaultsSuite) }
        let hash = String(repeating: "c", count: 64)
        let output = compactCheckpointJSON(id: hash, target: "normal", family: .compactNormal)
        let checkpoint = try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: output)
        let store = WorkbenchStore(preferences: defaults, workerOverride: { arguments, _ in
            XCTAssertEqual(arguments, ["checkpoint", "--checkpoint", checkpoint.checkpointPath])
            return output
        })
        var options = MaterialTrainingOptions()
        options.modelFamily = .compactNormal
        options.target = "normal"
        options.scope = "full-model"
        options.useWarmStart = true
        options.learningRate = 0.0007
        options.gradientAccumulationSteps = 5
        options.seed = 123
        options.validationEvery = 13
        options.validationUnit = .step
        options.checkpointEvery = 2
        options.checkpointUnit = .epoch
        options.loraRank = 0
        options.loraAlpha = 0
        let handoff = try MaterialTrainingHandoff(checkpoint: checkpoint, dataset: nil, training: options,
                                                 sampleID: nil, inputVariantID: nil)
        store.receiveTrainingHandoff(handoff)
        try await waitUntilIdle(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(store.training, options)
        XCTAssertEqual(store.selectedCheckpointId, hash)
        XCTAssertEqual(store.dependencyArguments(for: checkpoint), [])
    }

    func testLibraryRefinementAdoptsTheExactCheckpointAndOpensTrainer() throws {
        let defaults = try isolatedPreferences()
        defer { defaults.removePersistentDomain(forName: defaultsSuite) }
        let store = WorkbenchStore(preferences: defaults)
        let first = try checkpoint(id: "first", target: "height", scope: "final-map")
        let second = try checkpoint(id: "second", target: "normal", scope: "map-decoder")
        store.checkpoints = [first, second]
        store.selectedCheckpointId = first.id
        let request = store.trainingNavigationRequest
        XCTAssertTrue(store.selectCheckpointForRefinement(second))
        XCTAssertEqual(store.selectedCheckpointId, second.id)
        XCTAssertEqual(store.training.target, "normal")
        XCTAssertEqual(store.training.scope, "map-decoder")
        XCTAssertTrue(store.training.useWarmStart)
        XCTAssertNotEqual(store.trainingNavigationRequest, request)
    }

    func testComparisonUsesSelectedDiffuseVariantWhenPrimaryInputIsAbsent() throws {
        let defaults = try isolatedPreferences()
        defer { defaults.removePersistentDomain(forName: defaultsSuite) }
        let store = WorkbenchStore(preferences: defaults)
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: """
        {"dataset_path":"/dataset","index_sha256":"index","materials":[{"material_id":"soil","samples":[
          {"sample_id":"soil","status":"approved","split":"train","width":1024,"height":1024,"maps":{},
           "input_variants":[{"path":"/dataset/color.png","variant_id":"color"}]}]}]}
        """.replacingOccurrences(of: "\n", with: ""))
        store.selectedSampleId = "soil"
        store.selectedInputVariantId = "color"
        let model = try checkpoint(id: "model", target: "height", scope: "final-map")
        store.checkpoints = [model]
        store.comparisonCheckpointIds = [model.id]
        XCTAssertNil(store.comparisonConfigurationIssue)
        store.comparisonIncludesBase = false
        XCTAssertEqual(store.comparisonConfigurationIssue, "Include the base model or select a second checkpoint.")
    }

    func testDeveloperAutoUploadKeepsVisibilityPrivateUnlessChosen() throws {
        let global = StudioPreferences.defaults
        let saved = global.object(forKey: StudioPreferences.developerModeKey)
        defer { global.set(saved, forKey: StudioPreferences.developerModeKey) }
        global.set(true, forKey: StudioPreferences.developerModeKey)
        let defaults = try isolatedPreferences()
        defer { defaults.removePersistentDomain(forName: defaultsSuite) }
        let store = WorkbenchStore(preferences: defaults)
        XCTAssertFalse(store.uploadPublic)
        XCTAssertTrue(store.uploadAfterTraining)
        defaults.set(true, forKey: "uploadPublic")
        defaults.set(true, forKey: "uploadAfterTraining")
        let reopened = WorkbenchStore(preferences: defaults)
        XCTAssertTrue(reopened.uploadPublic)
        XCTAssertTrue(reopened.uploadAfterTraining)
    }

    func testOpeningMapsStartsACleanReviewSession() throws {
        let defaults = try isolatedPreferences()
        defer { defaults.removePersistentDomain(forName: defaultsSuite) }
        let review = ReviewSessionStore(preferences: defaults)
        review.manifestURL = URL(fileURLWithPath: "/previous/review.json")
        defaults.set(review.manifestURL!.path, forKey: "reviewManifest")
        review.blendURL = URL(fileURLWithPath: "/previous/scene.blend")
        review.decisions = ["old": "usable"]
        review.notes = ["old": "old note"]
        review.error = "old error"
        let fresh = URL(fileURLWithPath: "/new/raw.exr")
        review.openMaps([fresh])
        XCTAssertNil(review.manifestURL)
        XCTAssertNil(review.blendURL)
        XCTAssertNil(defaults.object(forKey: "reviewManifest"))
        XCTAssertTrue(review.decisions.isEmpty)
        XCTAssertTrue(review.notes.isEmpty)
        XCTAssertNil(review.error)
        XCTAssertEqual(review.selected?.candidates.first?.mapURL, fresh)
    }

    func testSavedDecisionsRestoreWithoutManuallyReopeningTheSavedFile() throws {
        let defaults = try isolatedPreferences()
        defer { defaults.removePersistentDomain(forName: defaultsSuite) }
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: folder) }
        let map = folder.appendingPathComponent("height.exr")
        let raw = Data("untouched numeric source".utf8)
        try raw.write(to: map)
        let review = ReviewSessionStore(preferences: defaults)
        review.openMaps([map])
        review.selectedCandidateId = map.path
        review.decisions[map.path] = "needs_work"
        review.notes[map.path] = "Inspect relief"
        let saved = folder.appendingPathComponent("decisions.json")
        try review.writeReview(to: saved)
        let reopened = ReviewSessionStore(preferences: defaults)
        reopened.restore(workspace: folder.path)
        XCTAssertNil(reopened.error)
        XCTAssertEqual(reopened.manifestURL, saved)
        XCTAssertEqual(reopened.decisions[map.path], "needs_work")
        XCTAssertEqual(reopened.notes[map.path], "Inspect relief")
        XCTAssertEqual(reopened.selectedCandidateId, map.path)
        XCTAssertEqual(try Data(contentsOf: map), raw)
        XCTAssertThrowsError(try review.writeReview(to: folder.appendingPathComponent("missing/decisions.json")))
        XCTAssertEqual(defaults.string(forKey: "reviewManifest"), saved.path)
    }

    func testAlreadyOpenTrainerQueuesAndUsesExactRequestedCheckpoint() async throws {
        let defaults = try isolatedPreferences()
        defer { defaults.removePersistentDomain(forName: defaultsSuite) }
        let hash = String(repeating: "a", count: 64)
        let selected = try checkpoint(id: hash, target: "normal", scope: "map-decoder")
        let output = checkpointJSON(id: hash, target: "normal", scope: "map-decoder")
        var calls = 0
        let store = WorkbenchStore(preferences: defaults, workerOverride: { args, _ in
            XCTAssertEqual(args, ["checkpoint", "--checkpoint", selected.checkpointPath])
            calls += 1
            return output
        })
        var options = MaterialTrainingOptions()
        options.useWarmStart = true
        options.target = "normal"
        options.scope = "map-decoder"
        options.validationEvery = 91
        let handoff = try MaterialTrainingHandoff(checkpoint: selected, dataset: nil, training: options,
                                                sampleID: nil, inputVariantID: nil)
        store.operation("Existing setup") { try await Task.sleep(for: .milliseconds(10)) }
        store.receiveTrainingHandoff(handoff)
        XCTAssertEqual(calls, 0, "Receiving a handoff cannot interrupt active work.")
        try await waitUntilIdle(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls, 1)
        XCTAssertEqual(store.selectedCheckpointId, hash)
        XCTAssertEqual(store.training.target, "normal")
        XCTAssertEqual(store.training.scope, "map-decoder")
        XCTAssertEqual(store.training.validationEvery, 91)
    }

    func testHandoffRejectsCheckpointWhoseBytesChanged() async throws {
        let defaults = try isolatedPreferences()
        defer { defaults.removePersistentDomain(forName: defaultsSuite) }
        let expected = try checkpoint(id: String(repeating: "a", count: 64), target: "normal", scope: "map-decoder")
        let changedOutput = checkpointJSON(id: String(repeating: "b", count: 64), target: "normal", scope: "map-decoder")
            .replacingOccurrences(of: "/runs/\(String(repeating: "b", count: 64))/", with: "/runs/\(String(repeating: "a", count: 64))/")
        let store = WorkbenchStore(preferences: defaults, workerOverride: { _, _ in changedOutput })
        var options = MaterialTrainingOptions()
        options.useWarmStart = true
        let handoff = try MaterialTrainingHandoff(checkpoint: expected, dataset: nil, training: options,
                                                sampleID: nil, inputVariantID: nil)
        store.receiveTrainingHandoff(handoff)
        try await waitUntilIdle(store)
        XCTAssertTrue(store.error?.contains("changed") == true)
        XCTAssertNil(store.selectedCheckpointId)
        XCTAssertFalse(store.training.useWarmStart)
    }

    private func waitUntilIdle(_ store: WorkbenchStore) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while store.isBusy, ContinuousClock.now < deadline { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(store.isBusy)
    }

    private var defaultsSuite = ""
    private func isolatedPreferences() throws -> UserDefaults {
        defaultsSuite = "org.ipde.interaction-tests.\(UUID().uuidString)"
        return try XCTUnwrap(UserDefaults(suiteName: defaultsSuite))
    }
    private func checkpoint(id: String, target: String, scope: String) throws -> WorkbenchCheckpoint {
        try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: checkpointJSON(id: id, target: target, scope: scope))
    }
    private func checkpointJSON(id: String, target: String, scope: String) -> String {
        """
        {"checkpoint_path":"/runs/\(id)/model.safetensors","sha256":"\(id)",
         "schema":"texture-studio-material-lora-v1","target":"\(target)","scope":"\(scope)",
         "step":42,"compatible":true,"supports_training_warm_start":true}
        """.replacingOccurrences(of: "\n", with: "")
    }
    private func compactCheckpointJSON(id: String, target: String, family: MaterialTrainingModelFamily) -> String {
        """
        {"checkpoint_path":"/runs/\(id)/model.safetensors","sha256":"\(id)",
         "schema":"texture-studio-compact-material-v1","target":"\(target)","scope":"full-model",
         "architecture":"\(family.architecture!)","variant":"full",
         "step":42,"compatible":true,"supports_training_warm_start":true}
        """.replacingOccurrences(of: "\n", with: "")
    }
}

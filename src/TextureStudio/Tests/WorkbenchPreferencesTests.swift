import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class WorkbenchPreferencesTests: XCTestCase {
    func testPreparationWorkersDefaultToAvailableCoresAndSaveImmediately() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let resources = MachineResources(physicalBytes: 32 * MachineResources.gibibyte, availableProcessorCount: 12)
        let first = WorkbenchStore(preferences: fixture.defaults, resources: resources)
        XCTAssertEqual(first.preparationWorkers, 12)
        first.preparationWorkers = 3
        let reopened = WorkbenchStore(preferences: fixture.defaults, resources: resources)
        XCTAssertEqual(reopened.preparationWorkers, 3)
        for value in [0, -1, Int.min] {
            reopened.preparationWorkers = value
            XCTAssertEqual(reopened.preparationWorkers, 1)
            XCTAssertEqual(WorkbenchStore(preferences: fixture.defaults, resources: resources).preparationWorkers, 1)
        }
    }

    func testPreparationWorkersAdaptToFewerAvailableCores() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        fixture.defaults.set(24, forKey: "preparationWorkers")
        let resources = MachineResources(physicalBytes: 16 * MachineResources.gibibyte, availableProcessorCount: 6)
        let store = WorkbenchStore(preferences: fixture.defaults, resources: resources)
        XCTAssertEqual(store.preparationWorkers, 6)
        for value in [100, Int.max] {
            store.preparationWorkers = value
            XCTAssertEqual(store.preparationWorkers, 6)
            XCTAssertEqual(WorkbenchStore(preferences: fixture.defaults, resources: resources).preparationWorkers, 6)
        }
    }

    func testFormAndComparisonChoicesSaveWithoutStartingWorkerOrClosingWindow() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let first = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        first.training.modelName = "Stone / warm evening"
        first.training.target = "normal"
        first.training.size = 2048
        first.training.updatesPerCrop = 875
        first.training.maxMinutes = 120
        first.training.useSelectedMaterialOnly = true
        first.training.useWarmStart = true
        first.selectedSampleId = "soil_crop_002"
        first.selectedRole = "roughness"
        first.selectedCheckpointId = "second-checkpoint"
        first.comparisonCheckpointIds = ["first-checkpoint", "second-checkpoint"]
        first.comparisonIncludesBase = false
        first.sourceImageURL = fixture.root.appendingPathComponent("source.png")
        first.lastPackageURL = fixture.root.appendingPathComponent("exported-model")
        first.lastPackageCheckpointId = "second-checkpoint"

        let reopened = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(reopened.training, first.training)
        XCTAssertEqual(reopened.training.modelName, "Stone / warm evening")
        XCTAssertEqual(reopened.selectedSampleId, "soil_crop_002")
        XCTAssertEqual(reopened.selectedRole, "roughness")
        XCTAssertEqual(reopened.selectedCheckpointId, "second-checkpoint")
        XCTAssertEqual(reopened.comparisonCheckpointIds, ["first-checkpoint", "second-checkpoint"])
        XCTAssertFalse(reopened.comparisonIncludesBase)
        XCTAssertEqual(reopened.sourceImageURL, first.sourceImageURL)
        XCTAssertEqual(reopened.lastPackageURL, first.lastPackageURL)
        XCTAssertEqual(reopened.lastPackageCheckpointId, "second-checkpoint")
    }

    func testRestoreKeepsSelectedCropAndCheckpointInsteadOfLastLoadedRow() async throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let dataset = fixture.root.appendingPathComponent("dataset.json")
        let firstPath = fixture.root.appendingPathComponent("first.pt")
        let secondPath = fixture.root.appendingPathComponent("second.pt")
        for url in [dataset, firstPath, secondPath] { try Data().write(to: url) }
        fixture.defaults.set(dataset.path, forKey: "dataset")
        fixture.defaults.set([firstPath.path, secondPath.path], forKey: "checkpoints")
        let original = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        original.selectedSampleId = "soil_002"
        original.selectedCheckpointId = "first"
        original.comparisonCheckpointIds = ["second"]
        original.comparisonIncludesBase = false
        original.training.size = 2048
        var calls: [String] = []
        let restored = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources, workerOverride: { arguments, _ in
            calls.append(arguments[0])
            if arguments.first == "capabilities" { return "{\"training_sizes\":[512,1024,2048]}" }
            if arguments.first == "dataset" { return try Self.datasetJSON() }
            if arguments.first == "prepare-size" { return try Self.datasetJSON(size: 2048, prepared: true) }
            let path = arguments[try XCTUnwrap(arguments.firstIndex(of: "--checkpoint")) + 1]
            let id = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
            return """
            {"checkpoint_path":"\(path)","sha256":"\(id)","schema":"texture-studio-material-lora-v1","target":"height","step":42,"compatible":true}
            """
        })
        restored.restore()
        try await settled(restored)
        XCTAssertNil(restored.error)
        XCTAssertEqual(restored.selectedSampleId, "soil_002")
        XCTAssertEqual(restored.selectedCheckpointId, "first")
        XCTAssertEqual(restored.comparisonCheckpointIds, ["second"])
        XCTAssertFalse(restored.comparisonIncludesBase)
        XCTAssertEqual(restored.training.size, 2048, "Reopening a 1K dataset cannot reset the requested 2K size")
        XCTAssertEqual(calls, ["capabilities", "dataset", "checkpoint", "checkpoint"])
        XCTAssertTrue(restored.dataset?.hasNativeSize(1024) == true, "Opening preserves source references and allocates no rescaled dataset")
        XCTAssertEqual(fixture.defaults.stringArray(forKey: "checkpoints"), [firstPath.path, secondPath.path])
        restored.restore()
        XCTAssertEqual(calls.count, 4, "A view appearing again must not reload and overwrite selections")
        let next = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(next.selectedCheckpointId, "first")
        XCTAssertEqual(next.selectedSampleId, "soil_002")
        XCTAssertEqual(next.comparisonCheckpointIds, ["second"])
    }

    func testCapabilityDiscoveryPreservesSupportedResolutionSelection() async throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let store = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources, workerOverride: { arguments, _ in
            return "{\"training_sizes\":[1024,2048]}"
        })
        store.training.size = 2048
        try await store.loadTrainingCapabilities()
        XCTAssertEqual(store.training.size, 2048)
        store.selectTrainingSize(1024)
        XCTAssertEqual(store.training.size, 1024)
        store.selectTrainingSize(2048)
        XCTAssertEqual(store.training.size, 2048)
        try await store.loadTrainingCapabilities()
        let reopened = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(reopened.training.size, 2048)
    }

    func testValidDatasetCanRunBeforeCapabilityDiscovery() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let resources = MachineResources(physicalBytes: 8 * MachineResources.gibibyte)
        let store = WorkbenchStore(preferences: fixture.defaults, resources: resources)
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: Self.datasetJSON())
        store.training.size = 1024
        XCTAssertNil(store.trainingConfigurationIssue)
        XCTAssertTrue(store.supportedTrainingSizes.contains(1024))
    }

    func testRuntimeAndUploadEditsSaveImmediately() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let first = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        first.workspacePath = fixture.root.path
        first.modelDirectory = "/located/model"
        first.uploadRepo = "artist/material-height"
        first.uploadPublic = true
        let reopened = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(reopened.workspacePath, fixture.root.path)
        XCTAssertEqual(reopened.modelDirectory, "/located/model")
        XCTAssertEqual(reopened.uploadRepo, "artist/material-height")
        XCTAssertTrue(reopened.uploadPublic)
    }

    func testPartialTrainingPreferencesKeepSupportedChoices() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let options = try JSONDecoder().decode(MaterialTrainingOptions.self,
            from: Data("{\"size\":2048}".utf8))
        XCTAssertEqual(options.size, 2048)
        XCTAssertEqual(options.modelName, "")
        XCTAssertEqual(options.target, "height")
        XCTAssertEqual(options.validationEvery, 1)
        XCTAssertEqual(options.validationUnit, .epoch)
        XCTAssertEqual(options.checkpointUnit, .epoch)
        XCTAssertEqual(options.restored(for: fixture.resources), options)
        fixture.defaults.set(Data("{\"training\":{\"size\":2048}}".utf8), forKey: WorkbenchPreferences.key)
        let restored = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(restored.training.size, 2048)
        XCTAssertTrue(restored.comparisonIncludesBase)
    }

    func testTrainingIntervalsDefaultToEpochsAndPersistIndependentUnits() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let first = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(first.training.validationEvery, 1)
        XCTAssertEqual(first.training.validationUnit, .epoch)
        XCTAssertEqual(first.training.checkpointEvery, 0)
        XCTAssertEqual(first.training.checkpointUnit, .epoch)
        first.training.validationEvery = 3
        first.training.validationUnit = .step
        first.training.checkpointEvery = 2
        first.training.checkpointUnit = .epoch
        let reopened = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(reopened.training, first.training)
        let data = try XCTUnwrap(fixture.defaults.data(forKey: WorkbenchPreferences.key))
        let preferences = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let training = try XCTUnwrap(preferences["training"] as? [String: Any])
        XCTAssertEqual(training["validationUnit"] as? String, "step")
        XCTAssertEqual(training["checkpointUnit"] as? String, "epoch")
    }

    func testLegacyTrainingIntervalsRemainStepsWithoutChangingCounts() throws {
        for document in ["{\"validationEvery\":91,\"checkpointEvery\":125}",
                         "{\"validationEvery\":0,\"checkpointEvery\":0}"] {
            let options = try JSONDecoder().decode(MaterialTrainingOptions.self, from: Data(document.utf8))
            XCTAssertEqual(options.validationUnit, .step)
            XCTAssertEqual(options.checkpointUnit, .step)
            let restored = try JSONDecoder().decode(MaterialTrainingOptions.self, from: JSONEncoder().encode(options))
            XCTAssertEqual(restored, options)
        }
        let partial = try JSONDecoder().decode(MaterialTrainingOptions.self, from: Data("{\"validationEvery\":91}".utf8))
        XCTAssertEqual(partial.validationEvery, 91)
        XCTAssertEqual(partial.validationUnit, .step)
        XCTAssertEqual(partial.checkpointUnit, .epoch)
        let explicit = try JSONDecoder().decode(MaterialTrainingOptions.self,
            from: Data("{\"validationEvery\":91,\"validationUnit\":\"epoch\",\"checkpointEvery\":125,\"checkpointUnit\":\"step\"}".utf8))
        XCTAssertEqual(explicit.validationEvery, 91)
        XCTAssertEqual(explicit.validationUnit, .epoch)
        XCTAssertEqual(explicit.checkpointEvery, 125)
        XCTAssertEqual(explicit.checkpointUnit, .step)
    }

    func testReopeningPreservesZeroQuickChecksAndValuesBeyondFormerUICaps() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        let first = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        first.training.validationEvery = 0
        first.training.checkpointEvery = 200_000
        first.training.updatesPerCrop = 50_000
        first.training.maxMinutes = 720
        first.training.loraRank = 128
        first.training.loraAlpha = 256
        XCTAssertEqual(WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources).training, first.training)
        first.training.validationEvery = Int.max
        first.training.maxMinutes = 0.1
        first.training.loraAlpha = 0.001
        XCTAssertEqual(WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources).training, first.training)
    }

    func testReopeningOnSmallerMacPreservesSettingsWithoutMemoryAdmissionAdaptation() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        var options = MaterialTrainingOptions()
        options.size = 2048
        options.updatesPerCrop = 800
        options.maxMinutes = 150
        WorkbenchPreferences(training: options).save(to: fixture.defaults)
        let smaller = MachineResources(physicalBytes: 16 * MachineResources.gibibyte)
        let restored = WorkbenchStore(preferences: fixture.defaults, resources: smaller)
        XCTAssertEqual(restored.training.size, 2048)
        XCTAssertEqual(restored.training.updatesPerCrop, 800)
        XCTAssertEqual(restored.training.maxMinutes, 150)
        XCTAssertTrue(restored.activity.isEmpty)
        fixture.defaults.set(Data("not JSON".utf8), forKey: WorkbenchPreferences.key)
        let fallback = WorkbenchStore(preferences: fixture.defaults, resources: fixture.resources)
        XCTAssertEqual(fallback.training.size, MaterialTrainingOptions().size)
    }

    func testStudioPreferencesRoundTripAndMissingFieldsKeepDefaultsAvailable() throws {
        let fixture = try PreferencesFixture()
        defer { fixture.remove() }
        var value = StudioPreferences()
        var settings = TextureSettings()
        settings.outputSize = 2048
        value.settings = settings
        value.depthChoice = .photoDetail
        value.modelID = "custom-local-model"
        value.customInverseDepth = false
        value.selectedPreview = "Normal"
        value.showInspector = false
        value.exportDirectory = fixture.root.path
        value.save(to: fixture.defaults)
        XCTAssertEqual(StudioPreferences.load(from: fixture.defaults), value)
        fixture.defaults.set(Data("{\"showInspector\":false}".utf8), forKey: StudioPreferences.key)
        let partial = StudioPreferences.load(from: fixture.defaults)
        XCTAssertEqual(partial.showInspector, false)
        XCTAssertNil(partial.settings)
        XCTAssertNil(partial.depthChoice)
    }

    private func settled(_ store: WorkbenchStore) async throws {
        let deadline = Date().addingTimeInterval(5)
        while store.isBusy, Date() < deadline { try await Task.sleep(for: .milliseconds(5)) }
        XCTAssertFalse(store.isBusy)
    }
    private static func datasetJSON(size: Int = 1024, prepared: Bool = false) throws -> String {
        let source = "/dataset/dataset.json"
        let path = prepared ? "/dataset/prepared-2048" : source
        let samples: [[String: Any]] = ["soil_001", "soil_002"].map { sample in
            ["sample_id": sample, "status": "approved", "split": "train", "width": size, "height": size,
             "maps": ["input": ["path": "/dataset/\(sample).png", "width": size, "height": size],
                      "height": ["path": "/dataset/\(sample)-height.png", "width": size, "height": size]]]
        }
        var document: [String: Any] = ["dataset_path": path, "index_sha256": prepared ? "prepared-sha" : "source-sha",
            "materials": [["material_id": "soil", "samples": samples]]]
        if prepared {
            document["automatic_validation"] = ["policy": "subject-extra-crops-v2", "material_ids": ["soil"]]
            document["preparation"] = ["source_dataset_path": source, "source_index_sha256": "source-sha",
                "prepared_dataset_path": path, "crop_size": size, "reused": true, "target_resized": false,
                "original_dataset_modified": false, "split_lineage_changed": true]
        }
        return String(decoding: try JSONSerialization.data(withJSONObject: document), as: UTF8.self)
    }

}

@MainActor private struct PreferencesFixture {
    let root: URL
    let defaults: UserDefaults
    let suite = "org.ipde.preference-tests.\(UUID().uuidString)"
    let resources = MachineResources(physicalBytes: 64 * MachineResources.gibibyte,
                                     metalRecommendedBytes: 48 * MachineResources.gibibyte)
    init() throws {
        root = FileManager.default.temporaryDirectory.appendingPathComponent("preference-tests-\(UUID().uuidString)")
        defaults = UserDefaults(suiteName: suite)!
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defaults.set(root.path, forKey: "workspace")
    }
    func remove() {
        defaults.removePersistentDomain(forName: suite)
        try? FileManager.default.removeItem(at: root)
    }
}

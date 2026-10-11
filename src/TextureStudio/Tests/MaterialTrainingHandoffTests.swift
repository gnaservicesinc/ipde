import Foundation
import XCTest
@testable import TextureStudio

final class MaterialTrainingHandoffTests: XCTestCase {
    func testRoundTripCapturesExactCheckpointDatasetSelectionAndTrainingSettings() throws {
        let handoff = try fixture()
        let document = try handoff.writeTemporary()
        defer { _ = try? handoff.discardTemporaryFile(at: document) }
        let restored = try MaterialTrainingHandoff.read(from: document)
        XCTAssertEqual(restored.schema, MaterialTrainingHandoff.schemaName)
        XCTAssertEqual(restored.requestID, handoff.requestID)
        XCTAssertEqual(restored.checkpointURL, URL(fileURLWithPath: "/models/chosen/adapter.safetensors"))
        XCTAssertEqual(restored.checkpointSHA256, String(repeating: "a", count: 64))
        XCTAssertEqual(restored.datasetURL, URL(fileURLWithPath: "/datasets/second/dataset.json"))
        XCTAssertEqual(restored.sampleID, "selected-material-crop")
        XCTAssertEqual(restored.inputVariantID, "col2")
        XCTAssertEqual(restored.training, handoff.training)
        XCTAssertEqual(restored.training.scope, "map-decoder")
        XCTAssertEqual(restored.training.target, "normal")
        XCTAssertEqual(restored.training.validationUnit, .step)
        XCTAssertEqual(restored.training.checkpointUnit, .epoch)
    }

    func testReadRejectsUnknownSchemaRemotePathsAndInvalidTrainingSettings() throws {
        let handoff = try fixture()
        let document = try handoff.writeTemporary()
        defer { _ = try? handoff.discardTemporaryFile(at: document) }
        let original = try Data(contentsOf: document)
        for mutation in ["schema", "checkpointURL", "checkpointSHA256", "datasetURL", "training"] {
            var payload = try XCTUnwrap(JSONSerialization.jsonObject(with: original) as? [String: Any])
            switch mutation {
            case "schema": payload[mutation] = "unknown-handoff-v2"
            case "checkpointURL", "datasetURL": payload[mutation] = "https://example.invalid/model.safetensors"
            case "checkpointSHA256": payload[mutation] = "changed"
            default:
                var training = try XCTUnwrap(payload[mutation] as? [String: Any])
                training["validationEvery"] = -1
                payload[mutation] = training
            }
            try JSONSerialization.data(withJSONObject: payload).write(to: document)
            XCTAssertThrowsError(try MaterialTrainingHandoff.read(from: document), mutation)
        }
    }

    func testHandoffPreservesDisabledQuickChecksAndUncappedTrainingSettings() throws {
        var training = options()
        training.validationEvery = 0
        training.checkpointEvery = 200_000
        training.updatesPerCrop = 50_000
        training.maxMinutes = 720
        training.loraRank = 128
        training.loraAlpha = 256
        let handoff = try MaterialTrainingHandoff(checkpoint: checkpoint(), dataset: nil, training: training,
                                                  sampleID: nil, inputVariantID: nil)
        let document = try handoff.writeTemporary()
        defer { _ = try? handoff.discardTemporaryFile(at: document) }
        XCTAssertEqual(try MaterialTrainingHandoff.read(from: document).training, training)
    }

    func testTemporaryCleanupKeepsSavedCopiesAndUnrelatedFiles() throws {
        let handoff = try fixture()
        let document = try handoff.writeTemporary()
        let folder = document.deletingLastPathComponent()
        let unrelated = folder.appendingPathComponent("keep.txt")
        try Data("keep".utf8).write(to: unrelated)
        let savedCopy = FileManager.default.temporaryDirectory.appendingPathComponent("saved-training-request-\(UUID()).json")
        defer {
            try? FileManager.default.removeItem(at: savedCopy)
            try? FileManager.default.removeItem(at: folder)
        }
        try handoff.write(to: savedCopy)
        XCTAssertFalse(try handoff.discardTemporaryFile(at: savedCopy))
        XCTAssertTrue(FileManager.default.fileExists(atPath: savedCopy.path))
        XCTAssertTrue(try handoff.discardTemporaryFile(at: document))
        XCTAssertFalse(FileManager.default.fileExists(atPath: document.path))
        XCTAssertEqual(try Data(contentsOf: unrelated), Data("keep".utf8))
    }

    private func fixture() throws -> MaterialTrainingHandoff {
        try MaterialTrainingHandoff(checkpoint: checkpoint(), dataset: URL(fileURLWithPath: "/datasets/second/dataset.json"),
                                    training: options(), sampleID: "selected-material-crop", inputVariantID: "col2")
    }
    private func options() -> MaterialTrainingOptions {
        var options = MaterialTrainingOptions()
        options.target = "normal"
        options.scope = "map-decoder"
        options.useWarmStart = true
        options.validationEvery = 91
        options.validationUnit = .step
        options.checkpointEvery = 125
        options.checkpointUnit = .epoch
        options.useSelectedMaterialOnly = true
        return options
    }
    private func checkpoint() throws -> WorkbenchCheckpoint {
        try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: """
        {"checkpoint_path":"/models/chosen/adapter.safetensors","sha256":"\(String(repeating: "A", count: 64))",
         "schema":"texture-studio-material-lora-v1","target":"normal","scope":"map-decoder",
         "step":42,"compatible":true,"supports_training_warm_start":true}
        """.replacingOccurrences(of: "\n", with: ""))
    }
}

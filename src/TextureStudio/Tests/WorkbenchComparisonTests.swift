import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class WorkbenchComparisonTests: XCTestCase {
    func testCompactComparisonRunsRecordedUntrainedInitializationWithoutPBRDependency() async throws {
        for target in ["height", "roughness", "normal"] {
            let root = try temporaryDirectory()
            defer { try? FileManager.default.removeItem(at: root) }
            let suite = "compact-comparison-\(UUID().uuidString)"
            let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
            defer { defaults.removePersistentDomain(forName: suite) }
            var calls: [[String]] = []
            let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { arguments, _ in
                calls.append(arguments)
                let output = arguments[try XCTUnwrap(arguments.firstIndex(of: "--output")) + 1]
                return "{\"outputs\":{\"\(target)\":{\"path\":\"\(output)/\(target).exr\"}},\"checkpoint_sha256\":\"exact-compact-sha\"}"
            })
            store.modelDirectory = root.appendingPathComponent("missing-pbr-base").path
            store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: """
            {"dataset_path":"/dataset","index_sha256":"dataset-sha","materials":[{"material_id":"soil","samples":[
              {"sample_id":"soil_crop","status":"approved","split":"train","width":1024,"height":1024,
               "maps":{"input":{"path":"/dataset/soil/photo.png"},"\(target)":{"path":"/dataset/soil/\(target).png"}}}]}]}
            """)
            store.selectedSampleId = "soil_crop"
            let family: MaterialTrainingModelFamily = target == "normal" ? .compactNormal : .compactScalar
            let checkpoint = try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: """
            {"checkpoint_path":"/models/compact/model.safetensors","sha256":"exact-compact-sha",
             "schema":"texture-studio-compact-material-v1","architecture":"\(family.architecture!)",
             "target":"\(target)","step":42,"compatible":true,"variant":"full","supports_training_warm_start":true}
            """)
            store.checkpoints = [checkpoint]
            store.comparisonCheckpointIds = [checkpoint.id]
            XCTAssertEqual(store.comparisonBaselineLabel, "Include untrained compact initialization")
            XCTAssertNil(store.comparisonConfigurationIssue)
            store.compare()
            try await waitForOperation(store)
            XCTAssertNil(store.error)
            XCTAssertEqual(calls.count, 2)
            XCTAssertTrue(calls[0].contains("--baseline"))
            XCTAssertFalse(calls[1].contains("--baseline"))
            for call in calls {
                XCTAssertFalse(call.contains("--model-directory"))
                XCTAssertTrue(call.contains("exact-compact-sha"))
                XCTAssertTrue(call.contains(checkpoint.checkpointPath))
            }
            let baseline = try XCTUnwrap(store.comparisonCandidates.first { $0.role == "base" })
            XCTAssertTrue(baseline.label.contains("Untrained initialization"))
            XCTAssertTrue(baseline.detail?.contains("recorded random initialization") == true)
            XCTAssertEqual(baseline.modelIdentity?.architecture, family.architecture)
            XCTAssertEqual(baseline.modelIdentity?.mapType, target)
            XCTAssertNil(baseline.modelIdentity?.checkpointStep)
        }
    }

    func testComparisonReconstructsTrainingGridAndRetainsOnlyOriginalReferences() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "comparison-grid-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(root.path, forKey: "workspace")
        let diffuse = root.appendingPathComponent("source-diffuse.png")
        let height = root.appendingPathComponent("source-height.png")
        try Data("original diffuse values".utf8).write(to: diffuse)
        try Data("original height values".utf8).write(to: height)
        let diffuseHash = ReviewImageLoader.hash(try Data(contentsOf: diffuse))
        let heightHash = ReviewImageLoader.hash(try Data(contentsOf: height))
        var calls: [[String]] = []
        var temporaryInput: URL?
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { arguments, _ in
            calls.append(arguments)
            let output = arguments[try XCTUnwrap(arguments.firstIndex(of: "--output")) + 1]
            if arguments.first == "review-source" {
                XCTAssertEqual(arguments[try XCTUnwrap(arguments.firstIndex(of: "--image")) + 1], diffuse.path)
                XCTAssertEqual(arguments[try XCTUnwrap(arguments.firstIndex(of: "--expected-sha256")) + 1], diffuseHash)
                XCTAssertEqual(arguments[try XCTUnwrap(arguments.firstIndex(of: "--size")) + 1], "1024")
                temporaryInput = URL(fileURLWithPath: output)
                try Data("exact training resize".utf8).write(to: temporaryInput!)
                return "{}"
            }
            let input = URL(fileURLWithPath: arguments[try XCTUnwrap(arguments.firstIndex(of: "--image")) + 1])
            XCTAssertEqual(input, temporaryInput)
            XCTAssertEqual(try Data(contentsOf: input), Data("exact training resize".utf8))
            return "{\"outputs\":{\"height\":{\"path\":\"\(output)/height.exr\"}},\"checkpoint_sha256\":\"exact-sha\"}"
        })
        let maps: [String: Any] = ["input": ["path": "/purged-stage/diffuse.png", "width": 1024, "height": 1024,
            "original_source_path": diffuse.path, "original_source_sha256": diffuseHash,
            "original_source_width": 4096, "original_source_height": 4096],
            "height": ["path": "/purged-stage/height.png", "width": 1024, "height": 1024,
            "source_bits": 16, "original_source_path": height.path, "original_source_sha256": heightHash,
            "original_source_width": 4096, "original_source_height": 4096]]
        let document: [String: Any] = ["dataset_path": root.path, "index_sha256": "index-sha", "materials": [
            ["material_id": "soil", "samples": [["sample_id": "soil_full", "status": "approved", "split": "train",
                "width": 1024, "height": 1024, "maps": maps]]]]]
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        store.dataset = try decoder.decode(WorkbenchDataset.self, from: JSONSerialization.data(withJSONObject: document))
        store.selectedSampleId = "soil_full"
        store.checkpoints = [try checkpoint(sha: "exact-sha", target: "height")]
        store.comparisonCheckpointIds = ["exact-sha"]
        store.compare()
        try await waitForOperation(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.map { $0[0] }, ["review-source", "infer", "infer"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: try XCTUnwrap(temporaryInput).deletingLastPathComponent().path))
        XCTAssertEqual(store.comparisonCandidates.first?.mapURL, diffuse)
        XCTAssertEqual(store.comparisonCandidates.first?.displayTransform?.size, 1024)
        XCTAssertEqual(store.comparisonCandidates.first { $0.role == "target" }?.mapURL, height)
        let review = ReviewSessionStore(preferences: defaults)
        review.load(try XCTUnwrap(store.lastOutputURL).appendingPathComponent("review-manifest.json"))
        XCTAssertNil(review.error)
        XCTAssertEqual(review.groups.first?.candidates.first?.mapURL, diffuse)
        XCTAssertEqual(review.groups.first?.candidates.first?.displayTransform?.sourceSHA256, diffuseHash)
        XCTAssertEqual(review.groups.first?.candidates.first { $0.role == "target" }?.displayTransform?.sourceSHA256, heightHash)
        XCTAssertEqual(try Data(contentsOf: diffuse), Data("original diffuse values".utf8))
    }

    func testOneCheckpointComparesWithHonestBaseAndLabelsSampleAndReference() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "comparison-test-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(root.path, forKey: "workspace")
        var calls: [[String]] = []
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { arguments, _ in
            calls.append(arguments)
            let output = arguments[try XCTUnwrap(arguments.firstIndex(of: "--output")) + 1]
            return "{\"outputs\":{\"height\":{\"path\":\"\(output)/height.exr\"}},\"checkpoint_sha256\":\"exact-sha\"}"
        })
        store.dataset = try WorkbenchResult.decode(WorkbenchDataset.self, output: """
        {"dataset_path":"/dataset","index_sha256":"dataset-sha","materials":[{"material_id":"soil","samples":[{"sample_id":"soil_crop_002","status":"approved","split":"train","width":2048,"height":2048,"maps":{"input":{"path":"/dataset/soil/photo.png"},"height":{"path":"/dataset/soil/displacement.png","source_bits":16}}}]}]}
        """)
        store.selectedSampleId = "soil_crop_002"
        store.checkpoints = [try checkpoint(sha: "exact-sha", target: "height")]
        store.comparisonCheckpointIds = ["exact-sha"]
        store.compare()
        try await waitForOperation(store)
        XCTAssertNil(store.error)
        XCTAssertEqual(calls.count, 2, "One baseline and one selected checkpoint run")
        XCTAssertTrue(calls[0].contains("--baseline"))
        XCTAssertFalse(calls[1].contains("--baseline"))
        for call in calls {
            XCTAssertEqual(call[try XCTUnwrap(call.firstIndex(of: "--image")) + 1], "/dataset/soil/photo.png")
            XCTAssertEqual(call[try XCTUnwrap(call.firstIndex(of: "--expected-sha256")) + 1], "exact-sha")
        }
        XCTAssertEqual(store.comparisonCandidates.map(\.role), ["diffuse", "target", "base", "checkpoint"])
        XCTAssertTrue(store.comparisonCandidates.allSatisfy { $0.sampleLabel == "soil_crop_002" })
        let base = try XCTUnwrap(store.comparisonCandidates.first { $0.role == "base" })
        XCTAssertTrue(base.label.contains("Base"))
        XCTAssertTrue(base.detail?.contains("before refinement") == true)
        XCTAssertEqual(base.modelIdentity?.architecture, "PBRnxt material model")
        XCTAssertNil(base.modelIdentity?.checkpointStep, "The untrained baseline does not inherit a trained checkpoint's step")
        let trained = try XCTUnwrap(store.comparisonCandidates.first { $0.role == "checkpoint" })
        XCTAssertTrue(trained.detail?.contains("step 42") == true)
        XCTAssertEqual(trained.modelIdentity?.checkpointPath, "/runs/material-2k/model.safetensors")
        XCTAssertEqual(trained.modelIdentity?.checkpointSHA256, "exact-sha")
        XCTAssertEqual(trained.modelIdentity?.checkpointStep, 42)
        XCTAssertEqual(trained.modelIdentity?.architecture, "PBRnxt material model")
        XCTAssertEqual(trained.modelIdentity?.mapType, "height")
        XCTAssertTrue(trained.accessibleLabel.contains("soil_crop_002"))
        XCTAssertTrue(trained.exportFilename.contains("soil_crop_002"))
        XCTAssertTrue(trained.exportFilename.contains("exact-sha"))
        XCTAssertFalse(trained.exportFilename.contains("/"))
        XCTAssertEqual(ReviewWorkbenchView.initialCandidates(store.comparisonCandidates).map(\.role), ["target", "base", "checkpoint"])
        let manifest = try XCTUnwrap(store.lastOutputURL).appendingPathComponent("review-manifest.json")
        let review = ReviewSessionStore()
        let savedPreference = UserDefaults(suiteName: "org.ipde.material-tools")!.object(forKey: "reviewManifest")
        defer { UserDefaults(suiteName: "org.ipde.material-tools")!.set(savedPreference, forKey: "reviewManifest") }
        review.load(manifest)
        XCTAssertNil(review.error)
        XCTAssertEqual(review.groups.first?.id, "soil_crop_002")
        XCTAssertEqual(review.groups.first?.candidates.map(\.role), ["diffuse", "target", "base", "checkpoint"])
        XCTAssertEqual(review.groups.first?.candidates.last?.detail, trained.detail)
        XCTAssertEqual(review.groups.first?.candidates.last?.modelIdentity, trained.modelIdentity)
        XCTAssertEqual(review.groups.first?.candidates.first { $0.role == "base" }?.modelIdentity, base.modelIdentity)
        let reviewed = root.appendingPathComponent("decisions.json")
        try review.writeReview(to: reviewed)
        review.load(reviewed)
        XCTAssertEqual(review.groups.first?.candidates.last?.modelIdentity, trained.modelIdentity,
            "Saving decisions must retain the inspected live model identity")
    }

    func testEverySelectedCheckpointIsInitiallyVisibleAndSourceCanBeEnabled() {
        let roles = ["source", "base", "checkpoint", "checkpoint", "checkpoint"]
        let candidates = roles.enumerated().map { index, role in
            MapReviewCandidate(id: "\(index)", label: "Candidate \(index)", mapURL: URL(fileURLWithPath: "/map-\(index).exr"),
                numeric: role != "source", sampleLabel: "sand_crop_003", detail: "Step \(index * 100)", role: role)
        }
        XCTAssertEqual(ReviewWorkbenchView.initialCandidates(candidates).map(\.id), ["1", "2", "3", "4"])
        XCTAssertTrue(candidates[0].accessibleLabel.contains("sand_crop_003"))
    }

    func testDisablingBaseStillRequiresTwoMatchingCheckpoints() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "comparison-test-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(root.path, forKey: "workspace")
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { _, _ in
            XCTFail("Invalid comparison must not launch workers")
            return "{}"
        })
        store.sourceImageURL = URL(fileURLWithPath: "/photo.png")
        store.checkpoints = [try checkpoint(sha: "height", target: "height"), try checkpoint(sha: "normal", target: "normal")]
        store.comparisonCheckpointIds = ["height"]
        store.comparisonIncludesBase = false
        store.compare()
        XCTAssertNotNil(store.error)
        XCTAssertFalse(store.isBusy)
        store.error = nil
        store.comparisonIncludesBase = true
        store.comparisonCheckpointIds = ["height", "normal"]
        store.compare()
        XCTAssertNotNil(store.error)
        XCTAssertFalse(store.isBusy)
    }

    func testComparisonUsesSelectedSamplesVerifiedFullSourceAndKeepsCropCoordinates() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let suite = "selected-reference-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        defaults.set(root.path, forKey: "workspace")
        var samples: [[String: Any]] = []
        var fullSources: [URL] = []
        var crops: [URL] = []
        for index in 0..<2 {
            let folder = root.appendingPathComponent("sample-\(index)")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            let map = folder.appendingPathComponent("displacement.png")
            let bytes = Data("Exact cropped height bytes \(index)".utf8)
            try bytes.write(to: map)
            let original = root.appendingPathComponent("original-4k-height-\(index).png")
            try Data("Full original height bytes \(index)".utf8).write(to: original)
            let hash = ReviewImageLoader.hash(bytes)
            let metadata: [String: Any] = ["sample_id": "sample-\(index)", "maps": ["height": "displacement.png"],
                "crop_rectangle_top_left_xywh": [index * 2048, 0, 2048, 2048],
                "map_metadata": ["height": ["sample_sha256": hash, "source": ["path": original.path,
                    "file_sha256": "original-source-sha-\(index)", "sample_bits": 16, "width": 4096, "height": 4096,
                    "published_url": "https://example.com/material-\(index)-disp.png"]]]]
            try JSONSerialization.data(withJSONObject: metadata).write(to: folder.appendingPathComponent("sample.json"))
            samples.append(["sample_id": "sample-\(index)", "status": "approved", "split": "train", "width": 2048, "height": 2048,
                "maps": ["input": ["path": folder.appendingPathComponent("photo.png").path],
                    "height": ["path": map.path, "sha256": hash, "source_bits": 16, "width": 2048, "height": 2048]]])
            fullSources.append(original); crops.append(map)
        }
        let document: [String: Any] = ["dataset_path": root.path, "index_sha256": "fixture-index-sha",
            "materials": [["material_id": "brick", "samples": samples]]]
        let store = WorkbenchStore(preferences: defaults, managedWorkspaceURL: root, workerOverride: { arguments, _ in
            let output = arguments[try XCTUnwrap(arguments.firstIndex(of: "--output")) + 1]
            return "{\"outputs\":{\"height\":{\"path\":\"\(output)/height.exr\"}},\"checkpoint_sha256\":\"exact-sha\"}"
        })
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        store.dataset = try decoder.decode(WorkbenchDataset.self, from: JSONSerialization.data(withJSONObject: document))
        store.selectedSampleId = "sample-1"
        store.checkpoints = [try checkpoint(sha: "exact-sha", target: "height")]
        store.comparisonCheckpointIds = ["exact-sha"]
        store.compare()
        // Selection may change while a comparison is running. The captured
        // sample, its source and the model input must remain the same pair.
        store.selectedSampleId = "sample-0"
        try await waitForOperation(store)
        XCTAssertNil(store.error)
        let reference = try XCTUnwrap(store.comparisonCandidates.first { $0.isReference })
        XCTAssertEqual(reference.sampleLabel, "sample-1")
        XCTAssertEqual(reference.mapURL, crops[1])
        XCTAssertEqual(reference.sourceIdentity?.path, fullSources[1].path)
        XCTAssertEqual(reference.sourceIdentity?.cropRectangle, [2048, 0, 2048, 2048])
        XCTAssertEqual(reference.sourceIdentity?.pixelDimensions, [4096, 4096])
        XCTAssertEqual(reference.fullSourceReference?.mapURL, fullSources[1])
        XCTAssertNotEqual(reference.fullSourceReference?.mapURL, fullSources[0])
        XCTAssertEqual(try Data(contentsOf: crops[1]), Data("Exact cropped height bytes 1".utf8))
    }

    private func checkpoint(sha: String, target: String) throws -> WorkbenchCheckpoint {
        try WorkbenchResult.decode(WorkbenchCheckpoint.self, output: """
        {"checkpoint_path":"/runs/material-2k/model.safetensors","sha256":"\(sha)","schema":"texture-studio-material-checkpoint-v1","target":"\(target)","step":42,"compatible":true}
        """)
    }

    private func waitForOperation(_ store: WorkbenchStore) async throws {
        let deadline = Date().addingTimeInterval(5)
        while store.isBusy, Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertFalse(store.isBusy)
    }

    private func temporaryDirectory() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("comparison-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return root
    }
}

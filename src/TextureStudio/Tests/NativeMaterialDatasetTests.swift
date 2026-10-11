import CryptoKit
import Darwin
import Foundation
import XCTest
@testable import TextureStudio

@MainActor
final class NativeMaterialDatasetTests: XCTestCase {
    func testSourceScanWorkersRespectCPUAndMappedFileMemory() {
        let gib = MachineResources.gibibyte
        let resources = MachineResources(physicalBytes: 16 * gib, availableProcessorCount: 12)
        XCTAssertEqual(NativeMaterialDatasetService.sourceScanWorkerCount(fileCount: 20, largestFileBytes: 1, resources: resources), 8)
        XCTAssertEqual(NativeMaterialDatasetService.sourceScanWorkerCount(fileCount: 3, largestFileBytes: 1, resources: resources), 3)
        XCTAssertEqual(NativeMaterialDatasetService.sourceScanWorkerCount(fileCount: 20, largestFileBytes: 256 * 1_048_576, resources: resources), 5)
        XCTAssertEqual(NativeMaterialDatasetService.sourceScanWorkerCount(fileCount: 20, largestFileBytes: UInt64.max, resources: resources), 1)
        XCTAssertEqual(NativeMaterialDatasetService.sourceScanWorkerCount(fileCount: 0, largestFileBytes: 0, resources: resources), 0)
        let singleCPU = MachineResources(physicalBytes: 16 * gib, availableProcessorCount: 1)
        XCTAssertEqual(NativeMaterialDatasetService.sourceScanWorkerCount(fileCount: 20, largestFileBytes: 1, resources: singleCPU), 1)
    }

    func testSourceScanReadsConcurrentlyPreservesOrderAndIsolatesCorruptFiles() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        var paths: [URL] = [], originals: [Data] = []
        for number in 0..<6 {
            let path = root.appendingPathComponent("map-\(number).png")
            let png = NativePNG(header: .init(width: 32, height: 32, bits: 8, channels: 3, color: 2, interlace: 0),
                                pixels: Data(repeating: UInt8(number), count: 32 * 32 * 3), colorChunks: [])
            var bytes = try png.encoded(); if number == 3 { bytes[29] ^= 1 }
            try bytes.write(to: path); paths.append(path); originals.append(bytes)
        }
        let inputPaths = paths, probe = SourceScanProbe()
        let results = try await Task.detached {
            try NativeMaterialDatasetService.scanSourceFiles(inputPaths, maximumWorkers: 2) { path in
                try probe.begin(requireConcurrentReaders: 2)
                defer { probe.end() }
                let bytes = try Data(contentsOf: path)
                _ = try NativePNG.sourceMetadata(bytes)
                return bytes
            }
        }.value
        XCTAssertEqual(probe.peakReaders, 2, "Two files must actually be inspected at the same time")
        XCTAssertEqual(probe.activeReaders, 0)
        XCTAssertEqual(probe.totalReaders, paths.count)
        XCTAssertEqual(results.count, paths.count)
        for (offset, result) in results.enumerated() {
            if offset == 3 { XCTAssertThrowsError(try result.get()) }
            else { XCTAssertEqual(try result.get(), originals[offset], "Completion order must not change file identity") }
            XCTAssertEqual(try Data(contentsOf: paths[offset]), originals[offset])
        }
    }

    func testCancellingSourceScanStopsAndJoinsReadersBeforeReturning() async throws {
        let paths = (0..<20).map { URL(fileURLWithPath: "/source-\($0).png") }, probe = SourceScanProbe()
        let task = Task.detached {
            try NativeMaterialDatasetService.scanSourceFiles(paths, maximumWorkers: 2) { _ in
                try probe.begin(requireConcurrentReaders: 2)
                defer { probe.end() }
                while true { try Task.checkCancellation(); Thread.sleep(forTimeInterval: 0.001) }
            }
        }
        XCTAssertTrue(probe.waitForReaders(2))
        task.cancel()
        do { _ = try await task.value; XCTFail("Cancelled scans must not return a partial inventory") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(probe.totalReaders, 2, "Cancellation must stop scheduling the remaining files")
        XCTAssertEqual(probe.activeReaders, 0, "The command must join every reader before returning")
    }

    func testParallelFolderScanPreservesNativeDimensionsVariantsAndImportProof() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Parallel scan", "--training-size", "256"])
        let rgb = try NativePNG(header: .init(width: 512, height: 512, bits: 8, channels: 3, color: 2, interlace: 0),
                                pixels: Data(repeating: 127, count: 512 * 512 * 3), colorChunks: []).encoded()
        let scalar = try NativePNG(header: .init(width: 512, height: 512, bits: 16, channels: 1, color: 0, interlace: 0),
                                   pixels: Data(repeating: 73, count: 512 * 512 * 2), colorChunks: []).encoded()
        for name in ["brick_diff_1k.png", "brick_diff2_1k.png", "brick_nor_dx_1k.png", "brick_nor_gl_1k.png"] {
            try rgb.write(to: sources.appendingPathComponent(name))
        }
        try scalar.write(to: sources.appendingPathComponent("brick_disp_1k.png"))
        try Data([1, 2, 3]).write(to: sources.appendingPathComponent("unrelated.png"))
        let inputName = "brick_diff_1k.png"
        let inputSHA = SHA256.hash(data: rgb).map { String(format: "%02x", $0) }.joined()
        let inputMD5 = Insecure.MD5.hash(data: rgb).map { String(format: "%02x", $0) }.joined()
        let provider: [String: Any] = ["material_id": "brick", "resolution": "1k", "provider": "Poly Haven", "license": "CC0-1.0",
            "api": ["api_url": "https://api.polyhaven.com/files/brick"], "completed_utc": "2026-10-09T00:00:00Z",
            "downloaded_maps": ["diff": ["path": sources.appendingPathComponent(inputName).path, "sha256": inputSHA, "bytes": rgb.count]],
            "maps": ["diff": ["published_md5": inputMD5, "published_bytes": rgb.count,
                              "url": "https://dl.polyhaven.org/file/ph-assets/Textures/png/1k/" + inputName]]]
        try JSONSerialization.data(withJSONObject: provider).write(to: sources.appendingPathComponent("material-source.json"))
        let firstPlan = root.appendingPathComponent("first.json"), secondPlan = root.appendingPathComponent("second.json")
        let first = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", firstPlan.path])
        let second = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", secondPlan.path])
        XCTAssertEqual(first["added_material_count"] as? Int, 1)
        XCTAssertEqual(first["ignored_file_count"] as? Int, 1)
        XCTAssertEqual(first["plan_sha256"] as? String, second["plan_sha256"] as? String, "Parallel scans must produce deterministic previews")
        let plan = try object(firstPlan), material = try XCTUnwrap((plan["materials"] as? [[String: Any]])?.first)
        XCTAssertEqual(material["common_pixel_dimensions"] as? [Int], [512, 512], "Actual pixels remain authoritative even when filenames say 1k")
        XCTAssertEqual((material["input_variants"] as? [[String: Any]])?.count, 2)
        let maps = try XCTUnwrap(material["maps"] as? [String: [String: Any]])
        XCTAssertEqual(maps["normal"]?["suffix"] as? String, "nor_gl")
        XCTAssertEqual(maps["input"]?["provider"] as? String, "Poly Haven")
        XCTAssertEqual(maps["input"]?["published_md5"] as? String, inputMD5)
        XCTAssertFalse(String(decoding: try Data(contentsOf: firstPlan), as: UTF8.self).contains("scan_verified_md5"),
                       "Temporary audit results must not leak into durable source metadata")
        let imported = try await output(["import-folder", "--dataset", dataset.path, "--folder", sources.path,
                                        "--plan", firstPlan.path, "--expected-plan-sha256", try XCTUnwrap(first["plan_sha256"] as? String)])
        XCTAssertEqual(imported["added_material_count"] as? Int, 1)
        XCTAssertEqual(try Data(contentsOf: sources.appendingPathComponent("brick_disp_1k.png")), scalar)
        XCTAssertEqual(try Data(contentsOf: sources.appendingPathComponent("brick_diff_1k.png")), rgb)
    }

    func testPreparationWorkersRespectCPUsAndWorkingMemory() throws {
        let gib = MachineResources.gibibyte
        let policy = NativeMaterialDatasetService.PreparationPolicy(resources: .init(physicalBytes: 8 * gib), processorCount: 12)
        XCTAssertEqual(policy.memoryBudgetBytes, gib)
        XCTAssertEqual(try policy.workerCount(jobCount: 20, estimatedPeakBytes: gib / 3), 3)
        XCTAssertEqual(try policy.workerCount(jobCount: 2, estimatedPeakBytes: gib / 3), 2)
        XCTAssertThrowsError(try policy.workerCount(jobCount: 2, estimatedPeakBytes: gib + 1))
        let cores = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: 8 * gib, maximumWorkers: 2)
        XCTAssertEqual(try cores.workerCount(jobCount: 8, estimatedPeakBytes: gib), 2)
        XCTAssertEqual(try policy.limitingWorkers(to: 2).workerCount(jobCount: 20, estimatedPeakBytes: gib / 3), 2)
        XCTAssertEqual(try policy.limitingWorkers(to: 100).maximumWorkers, 12)
        XCTAssertThrowsError(try policy.limitingWorkers(to: 0))
        let lowMemory = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: 32 * 1_048_576, maximumWorkers: 12)
        XCTAssertThrowsError(try lowMemory.workerCount(jobCount: 1, estimatedPeakBytes: 64 * 1_048_576))
        XCTAssertThrowsError(try lowMemory.limitingWorkers(to: 1).workerCount(jobCount: 1, estimatedPeakBytes: 64 * 1_048_576),
                             "Choosing one worker cannot admit a crop that exceeds the entire memory budget")
    }

    func testRecognizedPNGMapsWithWrongExtensionsImportAndPrepareWithoutChangingOriginals() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Misnamed PNG maps", "--training-size", "256"])
        let rgb = try fixturePNG(width: 512, height: 512, bits: 8, channels: 3)
        let heightPNG = try fixturePNG(width: 512, height: 512, bits: 16, channels: 1)
        let roughnessPNG = try fixturePNG(width: 512, height: 512, bits: 8, channels: 1)
        let layouts = [
            ("PolyHaven", "granite_tile_diff_1k.png", "granite_tile_disp_1k.txt", "granite_tile_rough_1k.txt"),
            ("AmbientCG", "Tiles014_1K-PNG_Color.txt", "Tiles014_1K-PNG_Displacement.txt", "Tiles014_1K-PNG_Roughness.png"),
            ("Generic", "diffuse.txt", "height.txt", "roughness.png")
        ]
        var originals: [URL: Data] = [:]
        for (directory, inputName, heightName, roughnessName) in layouts {
            let folder = sources.appendingPathComponent(directory)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            for (filename, bytes) in [(inputName, rgb), (heightName, heightPNG), (roughnessName, roughnessPNG)] {
                let file = folder.appendingPathComponent(filename); try bytes.write(to: file); originals[file] = bytes
            }
        }
        let canonicalHeight = sources.appendingPathComponent("PolyHaven/granite_tile_disp_1k.png")
        try heightPNG.write(to: canonicalHeight); originals[canonicalHeight] = heightPNG
        let planURL = root.appendingPathComponent("preview.json")
        let preview = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path])
        XCTAssertEqual(preview["source_set_count"] as? Int, 3)
        XCTAssertEqual(preview["added_material_count"] as? Int, 3)
        let notices = try XCTUnwrap(preview["warnings"] as? [String])
        for file in originals.keys where file.pathExtension == "txt" {
            XCTAssertTrue(notices.contains { $0.contains(file.path) && $0.contains(".txt extension") && $0.contains("detected and read as PNG") },
                          "Every misnamed PNG must identify its exact path and actual format, including identical duplicates")
        }
        let plan = try object(planURL), materials = try XCTUnwrap(plan["materials"] as? [[String: Any]])
        XCTAssertEqual(materials.count, 3)
        for material in materials {
            let maps = try XCTUnwrap(material["maps"] as? [String: [String: Any]])
            XCTAssertEqual(Set(maps.keys), ["input", "height", "roughness"])
            let heightPath = try XCTUnwrap(maps["height"]?["path"] as? String)
            if material["material_id"] as? String == "granite_tile_1k" {
                XCTAssertEqual(heightPath, canonicalHeight.path, "Identical target duplicates must keep one canonical source")
            } else { XCTAssertTrue(heightPath.hasSuffix(".txt"), "The original path must retain its actual extension") }
            XCTAssertEqual(maps["height"]?["sample_bits"] as? Int, 16)
        }
        _ = try await output(["import-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path,
                              "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String),
                              "--expected-index-sha256", try XCTUnwrap(preview["index_sha256"] as? String), "--training-size", "256"])
        let prepared = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"])
        let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height")
        XCTAssertEqual(descriptors.count, 3)
        let expected = try NativePNG.decode(heightPNG).crop([128, 128, 256, 256]).pixels
        for descriptor in descriptors { XCTAssertEqual(try NativePNG.decode(Data(contentsOf: descriptor.targetURL)).pixels, expected) }
        for (file, bytes) in originals { XCTAssertEqual(try Data(contentsOf: file), bytes) }
        _ = try await output(["cleanup-size", "--dataset", preparedURL.path])
    }

    func testMisnamedPNGMapChangeInvalidatesTheBoundImportPreview() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Bound import", "--training-size", "256"])
        let input = sources.appendingPathComponent("metal_plate_diff_1k.png"), height = sources.appendingPathComponent("metal_plate_disp_1k.txt")
        try fixturePNG(width: 256, height: 256, bits: 8, channels: 3).write(to: input)
        try fixturePNG(width: 256, height: 256, bits: 16, channels: 1).write(to: height)
        let planURL = root.appendingPathComponent("preview.json")
        let preview = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path])
        XCTAssertEqual(preview["added_material_count"] as? Int, 1)
        let originalIndex = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let replacement = try NativePNG(header: .init(width: 256, height: 256, bits: 16, channels: 1, color: 0, interlace: 0),
                                        pixels: Data(repeating: 9, count: 256 * 256 * 2), colorChunks: []).encoded()
        try replacement.write(to: height)
        do {
            _ = try await output(["import-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path,
                                  "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String)])
            XCTFail("Changing a misnamed original after preview must require another scan")
        } catch { XCTAssertTrue(error.localizedDescription.contains("changed")) }
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), originalIndex)
        XCTAssertEqual(try Data(contentsOf: height), replacement)
    }

    func testRecognizedNonPNGExtensionsStillRequireValidPNGContentAndChunkChecksums() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Invalid originals", "--training-size", "256"])
        let input = try fixturePNG(width: 256, height: 256, bits: 8, channels: 3)
        var damagedPNG = try fixturePNG(width: 256, height: 256, bits: 16, channels: 1)
        damagedPNG[29] ^= 1
        let invalidSources = [("text", Data("This is text, not a PNG image.".utf8)), ("crc", damagedPNG)]
        var originals: [URL: Data] = [:]
        for (name, bytes) in invalidSources {
            let folder = sources.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            try input.write(to: folder.appendingPathComponent(name + "_diff_1k.png"))
            let file = folder.appendingPathComponent(name + "_disp_1k.txt")
            try bytes.write(to: file); originals[file] = bytes
        }
        let preview = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path,
                                        "--plan", root.appendingPathComponent("preview.json").path])
        XCTAssertEqual(preview["source_set_count"] as? Int, 0)
        XCTAssertEqual(preview["added_material_count"] as? Int, 0)
        let notices = try XCTUnwrap(preview["warnings"] as? [String])
        XCTAssertTrue(notices.contains { $0.contains(sources.appendingPathComponent("crc/crc_disp_1k.txt").path) },
                      "A PNG signature alone must not bypass full chunk validation")
        XCTAssertTrue(notices.contains { $0.contains(sources.appendingPathComponent("text/text_disp_1k.txt").path) },
                      "Invalid content must identify the exact original file that was rejected")
        for (file, bytes) in originals { XCTAssertEqual(try Data(contentsOf: file), bytes) }
    }

    func testImportPreviewIdentifiesMissingAndUnsupportedTargetSources() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Useful preview", "--training-size", "256"])
        let fixtures = [("complete", 16), ("low_precision", 8), ("missing_height", 0)]
        var heightPaths: [String: String] = [:], inputPaths: [String: String] = [:]
        for (name, heightBits) in fixtures {
            let folder = sources.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            let input = folder.appendingPathComponent(name + "_diff_1k.png")
            try fixturePNG(width: 512, height: 512, bits: 8, channels: 3).write(to: input)
            inputPaths[name] = input.path
            if heightBits > 0 {
                let map = folder.appendingPathComponent(name + "_disp_1k.png")
                try fixturePNG(width: 512, height: 512, bits: heightBits, channels: 1).write(to: map)
                heightPaths[name] = map.path
            }
            try fixturePNG(width: 512, height: 512, bits: 8, channels: 1).write(to: folder.appendingPathComponent(name + "_rough_1k.png"))
            try fixturePNG(width: 512, height: 512, bits: 8, channels: 3).write(to: folder.appendingPathComponent(name + "_nor_gl_1k.png"))
        }
        let planURL = root.appendingPathComponent("preview.json")
        let preview = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path,
                                        "--plan", planURL.path])
        XCTAssertEqual(preview["source_set_count"] as? Int, 3)
        XCTAssertEqual(preview["warnings"] as? [String], [], "Nominal filename resolutions must not become per-map warning spam")
        let targets = try XCTUnwrap((preview["plans"] as? [String: [String: [String: Any]]])?["256"])
        let plan = try XCTUnwrap(targets["height"]), issues = try XCTUnwrap(plan["source_issues"] as? [[String: Any]])
        let missing = try XCTUnwrap(issues.first { $0["material_id"] as? String == "missing_height_1k" && $0["code"] as? String == "missing_target" })
        XCTAssertEqual(missing["source_path"] as? String, inputPaths["missing_height"])
        XCTAssertEqual(missing["target"] as? String, "height"); XCTAssertEqual(missing["crop_count"] as? Int, 1)
        XCTAssertFalse(try XCTUnwrap(missing["reason"] as? String).isEmpty)
        let lowPrecision = try XCTUnwrap(issues.first { $0["material_id"] as? String == "low_precision_1k" && $0["code"] as? String == "unsupported_precision" })
        XCTAssertEqual(lowPrecision["source_path"] as? String, heightPaths["low_precision"])
        XCTAssertEqual(lowPrecision["target"] as? String, "height"); XCTAssertEqual(lowPrecision["crop_count"] as? Int, 1)
        let precisionReason = try XCTUnwrap(lowPrecision["reason"] as? String)
        XCTAssertTrue(precisionReason.contains("8") && precisionReason.contains("16"))
        XCTAssertEqual(plan["train_count"] as? Int, 1)
        XCTAssertEqual(plan["unavailable_target_count"] as? Int, 2)
        let imported = try await output(["import-folder", "--dataset", dataset.path, "--folder", sources.path,
                                         "--plan", planURL.path, "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String)])
        XCTAssertEqual(imported["added_material_count"] as? Int, 3, "Missing optional targets must not prevent importing usable maps")
        for (target, expectedIDs) in [("height", ["complete_1k_center"]),
                                      ("roughness", ["complete_1k_center", "low_precision_1k_center", "missing_height_1k_center"]),
                                      ("normal", ["complete_1k_center", "low_precision_1k_center", "missing_height_1k_center"])] {
            let prepared = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", target])
            let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
            let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: target)
            XCTAssertEqual(Set(descriptors.map(\.id)), Set(expectedIDs), "Only the affected target may exclude a source set")
            _ = try await output(["cleanup-size", "--dataset", preparedURL.path])
        }
    }

    func testMixedResolutionImportSkipsSmallAlternatesAndKeepsTrueSizeDiagnostics() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Sources")
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Mixed resolutions", "--training-size", "256"])
        var originals: [URL: Data] = [:]
        let fixtures = [("Rock Face 04", "rock_face_04", "1k", 128),
                        ("Rock Face 04", "rock_face_04", "2k", 256),
                        ("Rock Face 04", "rock_face_04", "4k", 512),
                        ("Small only", "small_only", "1k", 128),
                        ("Small only", "unrelated", "2k", 256)]
        for (directory, family, label, dimension) in fixtures {
            let folder = sources.appendingPathComponent(directory)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            for (suffix, bits, channels) in [("diff", 8, 3), ("disp", 16, 1), ("rough", 8, 1), ("nor_gl", 8, 3)] {
                let path = folder.appendingPathComponent("\(family)_\(suffix)_\(label).png")
                let bytes = try fixturePNG(width: dimension, height: dimension, bits: bits, channels: channels)
                try bytes.write(to: path); originals[path] = bytes
            }
        }
        let planURL = root.appendingPathComponent("preview.json")
        let preview = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path])
        XCTAssertEqual(preview["source_set_count"] as? Int, 5)
        XCTAssertEqual(preview["warnings"] as? [String], [])
        let plans = try XCTUnwrap((preview["plans"] as? [String: [String: [String: Any]]])?["256"])
        for target in ["height", "roughness", "normal"] {
            let plan = try XCTUnwrap(plans[target])
            XCTAssertEqual(plan["train_count"] as? Int, 3)
            XCTAssertEqual(plan["undersized_source_set_count"] as? Int, 2, "Both smaller originals remain represented in the inventory")
            XCTAssertEqual(plan["smaller_alternate_source_set_count"] as? Int, 1)
            let issues = try XCTUnwrap(plan["source_issues"] as? [[String: Any]])
            XCTAssertEqual(issues.count, 1, "A usable sibling prevents a misleading small-alternate diagnostic")
            let issue = try XCTUnwrap(issues.first)
            XCTAssertEqual(issue["material_id"] as? String, "small_only_1k", "An unrelated family in the same folder cannot supply this material")
            XCTAssertEqual(issue["code"] as? String, "undersized")
            XCTAssertTrue(try XCTUnwrap(issue["reason"] as? String).contains("128 × 128"))
        }
        let imported = try await output(["import-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path,
                                         "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String), "--training-size", "256"])
        XCTAssertEqual(imported["added_material_count"] as? Int, 5, "The smaller originals must not fail or disappear during import")
        let entries = try XCTUnwrap(try object(dataset.appendingPathComponent("dataset.json"))["samples"] as? [[String: Any]])
        XCTAssertEqual(Set(entries.compactMap { $0["material_id"] as? String }),
                       ["rock_face_04_1k", "rock_face_04_2k", "rock_face_04_4k", "small_only_1k", "unrelated_2k"])
        for target in ["height", "roughness", "normal"] {
            let prepared = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", target])
            let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
            let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: target)
            XCTAssertEqual(Set(descriptors.map(\.id)), ["rock_face_04_2k_full", "rock_face_04_4k_center", "unrelated_2k_full"])
            for descriptor in descriptors {
                XCTAssertEqual(try NativePNG.decode(Data(contentsOf: descriptor.inputURL)).header.width, 256)
                XCTAssertEqual(try NativePNG.decode(Data(contentsOf: descriptor.targetURL)).header.height, 256)
            }
            _ = try await output(["cleanup-size", "--dataset", preparedURL.path])
        }
        for (path, bytes) in originals { XCTAssertEqual(try Data(contentsOf: path), bytes, "Import and preparation must preserve original pixels") }
    }

    func testLargerSiblingMissingTargetKeepsTargetExclusionInMixedResolutionPreview() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), sources = root.appendingPathComponent("Rock Face 04")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Target availability", "--training-size", "256"])
        for (label, dimension, suffix, bits, channels) in [("1k", 128, "diff", 8, 3), ("1k", 128, "disp", 16, 1),
                                                          ("2k", 256, "diff", 8, 3), ("2k", 256, "rough", 8, 1)] {
            try fixturePNG(width: dimension, height: dimension, bits: bits, channels: channels)
                .write(to: sources.appendingPathComponent("rock_face_04_\(suffix)_\(label).png"))
        }
        let preview = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path,
                                        "--plan", root.appendingPathComponent("preview.json").path])
        let plans = try XCTUnwrap((preview["plans"] as? [String: [String: [String: Any]]])?["256"])
        let height = try XCTUnwrap(plans["height"]), roughness = try XCTUnwrap(plans["roughness"])
        XCTAssertEqual(height["train_count"] as? Int, 0, "Larger pixels do not supply a missing training target")
        XCTAssertEqual(height["unavailable_target_count"] as? Int, 1)
        XCTAssertEqual(height["smaller_alternate_source_set_count"] as? Int, 1)
        let issues = try XCTUnwrap(height["source_issues"] as? [[String: Any]])
        XCTAssertEqual(issues.count, 1)
        XCTAssertEqual(issues.first?["material_id"] as? String, "rock_face_04_2k")
        XCTAssertEqual(issues.first?["code"] as? String, "missing_target")
        XCTAssertEqual(roughness["train_count"] as? Int, 1)
        XCTAssertEqual((roughness["source_issues"] as? [[String: Any]])?.count, 0)
    }

    func testPreparationCommandRespectsUserWorkerLimit() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await preparationFixture(root, materials: 4, dimension: 512)
        let policy = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: MachineResources.gibibyte, maximumWorkers: 4)
        let result = try await preparationOutput(["prepare-size", "--dataset", dataset.path, "--size", "256",
                                                  "--target", "height", "--preparation-workers", "2"], policy: policy)
        let prepared = URL(fileURLWithPath: try XCTUnwrap(result["dataset_path"] as? String))
        let index = try object(prepared.appendingPathComponent("dataset.json"))
        let binding = try XCTUnwrap(index["native_size_preparation"] as? [String: Any])
        XCTAssertEqual(binding["preparation_workers"] as? Int, 2)
    }

    func testPreparedReuseSkipsUnchangedFileChecksumsAndDetectsChangedBytes() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        for dimension in [256, 512] {
            let fixtureRoot = root.appendingPathComponent("Grid-\(dimension)")
            try FileManager.default.createDirectory(at: fixtureRoot, withIntermediateDirectories: false)
            let dataset = try await preparationFixture(fixtureRoot, materials: 1, dimension: dimension)
            let arguments = ["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"]
            let prepared = try await output(arguments), preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
            let index = try object(preparedURL.appendingPathComponent("dataset.json"))
            let binding = try XCTUnwrap(index["native_size_preparation"] as? [String: Any]), proof = try XCTUnwrap(binding["source_snapshot"] as? [String: Any])
            let descriptor = try XCTUnwrap(NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height").first)
            let originalInput = try Data(contentsOf: descriptor.inputURL)
            var reads: [URL] = []
            func checksum(_ file: URL) throws -> String {
                reads.append(file)
                return SHA256.hash(data: try Data(contentsOf: file, options: .mappedIfSafe)).map { String(format: "%02x", $0) }.joined()
            }
            XCTAssertTrue(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof, checksumFile: checksum))
            XCTAssertTrue(reads.isEmpty, "Both exact original references and generated crops reuse their recorded file state")
            let entry = try XCTUnwrap((index["samples"] as? [[String: Any]])?.first)
            let sampleURL = preparedURL.appendingPathComponent(try XCTUnwrap(entry["path"] as? String)).appendingPathComponent("sample.json")
            var legacy = try object(sampleURL), maps = try XCTUnwrap(legacy["map_metadata"] as? [String: [String: Any]])
            var input = try XCTUnwrap(maps["input"])
            if dimension == 256 {
                var source = try XCTUnwrap(input["source"] as? [String: Any]); source.removeValue(forKey: "source_stat"); input["source"] = source
            } else { input.removeValue(forKey: "prepared_stat") }
            maps["input"] = input; legacy["map_metadata"] = maps
            try JSONSerialization.data(withJSONObject: legacy).write(to: sampleURL)
            XCTAssertTrue(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof, checksumFile: checksum))
            XCTAssertEqual(reads, [descriptor.inputURL], "Older references or crops without stored state receive one full checksum check")
            reads.removeAll()
            XCTAssertTrue(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof, checksumFile: checksum))
            XCTAssertTrue(reads.isEmpty, "The missing state is restored after verification")
            let attributes = try FileManager.default.attributesOfItem(atPath: descriptor.inputURL.path)
            let modified = try XCTUnwrap(attributes[.modificationDate] as? Date)
            try FileManager.default.setAttributes([.modificationDate: modified.addingTimeInterval(2)], ofItemAtPath: descriptor.inputURL.path)
            XCTAssertTrue(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof, checksumFile: checksum))
            XCTAssertEqual(reads, [descriptor.inputURL], "A changed stat triggers one checksum, even for the duplicated canonical input variant")
            reads.removeAll()
            XCTAssertTrue(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof, checksumFile: checksum))
            XCTAssertTrue(reads.isEmpty, "Verified timestamps are refreshed so subsequent reuse remains cheap")
            var damaged = originalInput; damaged[damaged.count - 13] ^= 1
            try damaged.write(to: descriptor.inputURL)
            reads.removeAll()
            XCTAssertFalse(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof, checksumFile: checksum))
            XCTAssertEqual(reads, [descriptor.inputURL], "Corruption after the IHDR must still fail checksum validation")
            try originalInput.write(to: descriptor.inputURL)
            XCTAssertTrue(try NativeMaterialDatasetService.validatePrepared(preparedURL, size: 256, proof: proof))
            let reused = try await output(arguments)
            XCTAssertEqual((reused["preparation"] as? [String: Any])?["reused"] as? Bool, true)
            _ = try await output(["cleanup-size", "--dataset", preparedURL.path])
        }
    }

    func testStagedMetadataJournalRejectsCorruptionAndRecoversAlreadyPublishedReplacement() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Recovery", "--training-size", "256"])
        let indexURL = dataset.appendingPathComponent("dataset.json"), original = try Data(contentsOf: indexURL)
        var replacement = try object(indexURL); replacement["description"] = "Recovered replacement"
        let planned = try JSONSerialization.data(withJSONObject: replacement, options: [.sortedKeys])
        func sha(_ bytes: Data) -> String { SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined() }
        let stageName = ".material-workbench-transaction-\(UUID().uuidString)", stage = dataset.appendingPathComponent(stageName)
        try FileManager.default.createDirectory(at: stage, withIntermediateDirectories: false)
        let stagedURL = stage.appendingPathComponent("0.json"), journalURL = dataset.appendingPathComponent(".material-workbench-journal.json")
        let journal: [String: Any] = ["schema": "texture-studio-material-workbench-v1", "changes": [["path": "dataset.json", "old_sha256": sha(original), "new_sha256": sha(planned), "staged_path": stageName + "/0.json"]]]
        try JSONSerialization.data(withJSONObject: journal).write(to: journalURL)
        try Data("damaged".utf8).write(to: stagedURL)
        do { _ = try await output(["dataset", "--dataset", dataset.path]); XCTFail("Changed recovery bytes must not publish") }
        catch { XCTAssertTrue(error.localizedDescription.contains("recovery identity changed")) }
        XCTAssertEqual(try Data(contentsOf: indexURL), original)
        XCTAssertTrue(FileManager.default.fileExists(atPath: journalURL.path))
        try planned.write(to: stagedURL)
        // A crash after publication leaves the new revision at its destination.
        // Replaying the unchanged staged replacement must remain safe.
        try planned.write(to: indexURL)
        let recovered = try await output(["dataset", "--dataset", dataset.path])
        XCTAssertEqual(recovered["description"] as? String, "Recovered replacement")
        XCTAssertEqual(try Data(contentsOf: indexURL), planned)
        XCTAssertFalse(FileManager.default.fileExists(atPath: journalURL.path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: stage.path))
    }

    func testLargeSourcePreparationBudgetIncludesMappedInputAndAllRetainedCrops() throws {
        let gib = MachineResources.gibibyte
        let source: [String: Any] = ["width": 8192, "height": 8192, "channels": 3, "sample_bits": 16, "file_bytes": 363_824_405]
        let peak = try NativeMaterialDatasetService.preparationPeakBytes(sources: [source], size: 2048, cropCount: 3)
        // Covers an actual 8K RGB16 map plus three 2K crops and encoder buffers.
        XCTAssertGreaterThan(peak, 363_824_405 + 3 * 2048 * 2048 * 6)
        XCTAssertLessThan(peak, 700 * 1024 * 1024)
        let policy = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: gib, maximumWorkers: 128)
        XCTAssertEqual(try policy.workerCount(jobCount: 1000, estimatedPeakBytes: peak), 1)
        let tiny = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: 128 * 1024 * 1024, maximumWorkers: 128)
        XCTAssertThrowsError(try tiny.workerCount(jobCount: 1000, estimatedPeakBytes: peak))
        XCTAssertThrowsError(try NativeMaterialDatasetService.preparationPeakBytes(sources: [["width": -1]], size: 2048, cropCount: 3))
    }

    func testPreparationExportsOnlySelectedTargetAndDoesNotReadUnrelatedMaps() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Target scope", "--training-size", "256"])
        var paths: [String: URL] = [:]
        for (role, channels, bits) in [("input", 3, 8), ("height", 1, 16), ("roughness", 1, 8), ("normal", 3, 8)] {
            let url = root.appendingPathComponent(role + ".png"), color: UInt8 = channels == 1 ? 0 : 2
            try NativePNG(header: .init(width: 512, height: 512, bits: bits, channels: channels, color: color, interlace: 0),
                          pixels: Data(repeating: 97, count: 512 * 512 * channels * bits / 8), colorChunks: []).encoded().write(to: url)
            paths[role] = url
        }
        _ = try await output(["add-material", "--dataset", dataset.path, "--name", "Target scope"] + paths.keys.sorted().flatMap { ["--" + $0, paths[$0]!.path] })
        let originalIndex = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        for target in ["roughness", "normal", "height"] {
            // Displacement must continue working when irrelevant originals
            // cannot even be opened, rather than merely omitting their export.
            if target == "height" { for role in ["roughness", "normal"] { try FileManager.default.removeItem(at: paths[role]!) } }
            let result = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", target])
            let prepared = URL(fileURLWithPath: try XCTUnwrap(result["dataset_path"] as? String))
            let index = try object(prepared.appendingPathComponent("dataset.json"))
            XCTAssertEqual((index["native_size_preparation"] as? [String: Any])?["target"] as? String, target)
            for entry in try XCTUnwrap(index["samples"] as? [[String: Any]]) {
                let folder = prepared.appendingPathComponent(try XCTUnwrap(entry["path"] as? String))
                let metadata = try object(folder.appendingPathComponent("sample.json"))
                XCTAssertEqual(Set((metadata["maps"] as? [String: String] ?? [:]).keys), ["input", target])
                XCTAssertEqual(metadata["available_targets"] as? [String], [target])
                XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: folder.path).filter { $0.hasSuffix(".png") }.count, 2)
                XCTAssertNil(metadata["source_files"])
            }
            XCTAssertFalse(try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: target).isEmpty)
            let absent = ["height", "roughness", "normal"].first { $0 != target }!
            XCTAssertTrue(try NativeMaterialDatasetService.trainingSamples(datasetURL: prepared, size: 256, target: absent).isEmpty)
            _ = try await output(["cleanup-size", "--dataset", prepared.path])
        }
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), originalIndex)
    }

    func testColorProfilesStayInPNGAndLegacyMetadataCompactsWithoutChangingReviewBinding() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset"), input = root.appendingPathComponent("input.png"), height = root.appendingPathComponent("height.png")
        let profile = Data(repeating: 79, count: 2 * 1024 * 1024)
        let rgb = NativePNG(header: .init(width: 512, height: 512, bits: 8, channels: 3, color: 2, interlace: 0),
                            pixels: Data(repeating: 97, count: 512 * 512 * 3), colorChunks: [("iCCP", profile)])
        let bytes = try rgb.encoded(); try bytes.write(to: input)
        let summary = try NativePNG.sourceMetadata(bytes)
        XCTAssertNil(summary["color_chunks"])
        XCTAssertLessThan(try JSONSerialization.data(withJSONObject: summary).count, 512)
        try NativePNG(header: .init(width: 512, height: 512, bits: 16, channels: 1, color: 0, interlace: 0), pixels: Data(repeating: 97, count: 512 * 512 * 2), colorChunks: []).encoded().write(to: height)
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Compact profiles", "--training-size", "256"])
        let imported = try await output(["add-material", "--dataset", dataset.path, "--name", "Compact profiles", "--input", input.path, "--height", height.path])
        let index = try object(dataset.appendingPathComponent("dataset.json")), entry = try XCTUnwrap((index["samples"] as? [[String: Any]])?.first)
        let sampleURL = dataset.appendingPathComponent(try XCTUnwrap(entry["path"] as? String)).appendingPathComponent("sample.json")
        XCTAssertLessThan(try Data(contentsOf: sampleURL).count, 16 * 1024)
        let material = try XCTUnwrap((imported["materials"] as? [[String: Any]])?.first), sample = try XCTUnwrap((material["samples"] as? [[String: Any]])?.first)
        let sampleID = try XCTUnwrap(sample["sample_id"] as? String)
        _ = try await output(["curate", "--dataset", dataset.path, "--sample", sampleID, "--review-size", "256", "--status", "approved", "--note", "Keep profile"])
        var legacy = try object(sampleURL), maps = try XCTUnwrap(legacy["map_metadata"] as? [String: [String: Any]])
        var details = try XCTUnwrap(maps["input"]), source = try XCTUnwrap(details["source"] as? [String: Any])
        source["color_chunks"] = [["type": "iCCP", "data_base64": profile.base64EncodedString()]]; source["file_md5"] = "redundant"
        details["source"] = source; maps["input"] = details; legacy["map_metadata"] = maps; legacy["source_files"] = [source]
        try JSONSerialization.data(withJSONObject: legacy).write(to: sampleURL)
        let reopened = try await output(["dataset", "--dataset", dataset.path])
        let compactBytes = try Data(contentsOf: sampleURL), text = String(decoding: compactBytes, as: UTF8.self)
        XCTAssertLessThan(compactBytes.count, 16 * 1024); XCTAssertFalse(text.contains("base64")); XCTAssertFalse(text.contains("file_md5"))
        let reopenedMaterial = try XCTUnwrap((reopened["materials"] as? [[String: Any]])?.first), reopenedSample = try XCTUnwrap((reopenedMaterial["samples"] as? [[String: Any]])?.first)
        XCTAssertEqual(reopenedSample["status"] as? String, "approved"); XCTAssertEqual(reopenedSample["note"] as? String, "Keep profile")
        let prepared = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"])
        let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String)), descriptor = try XCTUnwrap(NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height").first)
        XCTAssertEqual(try NativePNG.decode(Data(contentsOf: descriptor.inputURL)).colorChunks.first?.1, profile)
        XCTAssertEqual(try Data(contentsOf: input), bytes)
        _ = try await output(["cleanup-size", "--dataset", preparedURL.path])
        XCTAssertFalse(try FileManager.default.contentsOfDirectory(atPath: dataset.path).contains { $0.hasPrefix(".material-workbench-transaction-") })
    }

    func testPreparationKeepsByteIdenticalColorVariantsAtDistinctPaths() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Bricks"), dataset = root.appendingPathComponent("Dataset")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let canonical = sources.appendingPathComponent("brick_diff_1k.png")
        let alternate = sources.appendingPathComponent("brick_diff2_1k.png")
        let height = sources.appendingPathComponent("brick_disp_1k.png")
        let pixels = Data((0..<512 * 512 * 3).map { UInt8(truncatingIfNeeded: $0 * 97) })
        let rgb = NativePNG(header: .init(width: 512, height: 512, bits: 8, channels: 3, color: 2, interlace: 0), pixels: pixels, colorChunks: [])
        let inputBytes = try rgb.encoded()
        let heightBytes = try NativePNG(header: .init(width: 512, height: 512, bits: 16, channels: 1, color: 0, interlace: 0),
                                        pixels: Data(repeating: 79, count: 512 * 512 * 2), colorChunks: []).encoded()
        try inputBytes.write(to: canonical); try inputBytes.write(to: alternate); try heightBytes.write(to: height)
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Identical variants", "--training-size", "256"])
        let plan = root.appendingPathComponent("preview.json")
        let preview = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", plan.path])
        XCTAssertEqual(preview["added_material_count"] as? Int, 1)
        _ = try await output(["import-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", plan.path,
                              "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String),
                              "--expected-index-sha256", try XCTUnwrap(preview["index_sha256"] as? String), "--training-size", "256"])
        let originalIndex = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let index = try object(dataset.appendingPathComponent("dataset.json"))
        let entry = try XCTUnwrap((index["samples"] as? [[String: Any]])?.first)
        let sourceSampleURL = dataset.appendingPathComponent(try XCTUnwrap(entry["path"] as? String)).appendingPathComponent("sample.json")
        var sourceSample = try object(sourceSampleURL)
        let originalVariants = try XCTUnwrap(sourceSample["input_variants"] as? [[String: Any]])
        XCTAssertEqual(originalVariants.count, 2)
        // Move the noncanonical source first: hash equality or list order must
        // never replace the explicitly recorded canonical input source path.
        sourceSample["input_variants"] = Array(originalVariants.reversed())
        let sourceMetadata = try JSONSerialization.data(withJSONObject: sourceSample, options: [.sortedKeys])
        try sourceMetadata.write(to: sourceSampleURL)
        let prepared = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"])
        let preparedURL = URL(fileURLWithPath: try XCTUnwrap(prepared["dataset_path"] as? String))
        let preparedIndex = try object(preparedURL.appendingPathComponent("dataset.json"))
        let records = try XCTUnwrap(preparedIndex["samples"] as? [[String: Any]])
        XCTAssertFalse(records.isEmpty)
        let expectedPixels = try rgb.crop([128, 128, 256, 256]).pixels
        for record in records {
            let folder = preparedURL.appendingPathComponent(try XCTUnwrap(record["path"] as? String))
            let sample = try object(folder.appendingPathComponent("sample.json"))
            let variants = try XCTUnwrap(sample["input_variants"] as? [[String: Any]])
            let names = variants.compactMap { $0["filename"] as? String }
            XCTAssertEqual(names, ["input-variant-1.png", "diffuse.png"])
            XCTAssertEqual(Set(names).count, 2)
            XCTAssertEqual(variants.compactMap { $0["variant_id"] as? String }, ["color_2", "color_default"])
            XCTAssertEqual(((variants[0]["source"] as? [String: Any])?["path"] as? String), alternate.resolvingSymlinksInPath().path)
            let maps = try XCTUnwrap(sample["map_metadata"] as? [String: [String: Any]])
            XCTAssertEqual(((maps["input"]?["source"] as? [String: Any])?["path"] as? String), canonical.resolvingSymlinksInPath().path)
            XCTAssertEqual((sample["maps"] as? [String: String])?["input"], "diffuse.png")
            for name in names { XCTAssertEqual(try NativePNG.decode(Data(contentsOf: folder.appendingPathComponent(name))).pixels, expectedPixels) }
        }
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height")
        XCTAssertEqual(descriptors.count, records.count * 2)
        XCTAssertEqual(Set(descriptors.map { $0.inputURL.path }).count, records.count * 2)
        XCTAssertEqual(try Data(contentsOf: canonical), inputBytes); XCTAssertEqual(try Data(contentsOf: alternate), inputBytes)
        XCTAssertEqual(try Data(contentsOf: height), heightBytes)
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), originalIndex)
        XCTAssertEqual(try Data(contentsOf: sourceSampleURL), sourceMetadata)
        _ = try await output(["cleanup-size", "--dataset", preparedURL.path])
    }

    func testParallelPreparationKeepsOrderedMetadataAndExactCropBytes() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await preparationFixture(root, materials: 4, dimension: 512)
        let originalIndex = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let policy = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: MachineResources.gibibyte, maximumWorkers: 1)
        let arguments = ["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"]
        let serial = try await preparationOutput(arguments, policy: policy)
        let serialRoot = URL(fileURLWithPath: try XCTUnwrap(serial["dataset_path"] as? String))
        let firstInventory = try preparedInventory(serialRoot)
        let firstIndex = try object(serialRoot.appendingPathComponent("dataset.json"))
        XCTAssertEqual((firstIndex["native_size_preparation"] as? [String: Any])?["preparation_workers"] as? Int, 1)
        _ = try await output(["cleanup-size", "--dataset", serialRoot.path])
        // Restore identical review inputs; cleanup deliberately persists the
        // generated crop review snapshot as a new per-size sidecar.
        try FileManager.default.removeItem(at: dataset.appendingPathComponent(".material-size-reviews.json"))
        let concurrent = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: MachineResources.gibibyte, maximumWorkers: 4)
        let events = PreparationEventRecorder()
        let parallel = try await preparationOutput(arguments, policy: concurrent, events: events)
        let parallelRoot = URL(fileURLWithPath: try XCTUnwrap(parallel["dataset_path"] as? String))
        let secondIndex = try object(parallelRoot.appendingPathComponent("dataset.json"))
        let binding = try XCTUnwrap(secondIndex["native_size_preparation"] as? [String: Any])
        XCTAssertEqual(binding["preparation_workers"] as? Int, 4)
        let peak = try XCTUnwrap(binding["estimated_worker_peak_bytes"] as? NSNumber).uint64Value
        XCTAssertLessThanOrEqual(peak * 4, concurrent.memoryBudgetBytes)
        XCTAssertEqual(try preparedInventory(parallelRoot), firstInventory)
        XCTAssertEqual(try JSONSerialization.data(withJSONObject: firstIndex["samples"]!, options: [.sortedKeys]),
                       try JSONSerialization.data(withJSONObject: secondIndex["samples"]!, options: [.sortedKeys]))
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), originalIndex)
        let recorded = try events.objects()
        XCTAssertEqual(recorded.first?["event"] as? String, "preparation_started")
        XCTAssertEqual(recorded.last?["event"] as? String, "preparation_completed")
        XCTAssertEqual(recorded.last?["completed"] as? Int, 4)
        XCTAssertTrue(recorded.allSatisfy { $0["worker_count"] as? Int == 4 && $0["total"] as? Int == 4 })
        let counts = recorded.compactMap { $0["completed"] as? Int }
        XCTAssertEqual(counts, counts.sorted())
        XCTAssertTrue(recorded.contains { $0["event"] as? String == "preparation_progress" })
        let samples = try NativeMaterialDatasetService.trainingSamples(datasetURL: parallelRoot, size: 256, target: "height")
        XCTAssertEqual(samples.count, 4)
        for sample in samples {
            let details = try object(parallelRoot.appendingPathComponent("samples/" + sample.id + "/sample.json"))
            let maps = try XCTUnwrap(details["map_metadata"] as? [String: [String: Any]])
            let source = try XCTUnwrap(maps["height"]?["source"] as? [String: Any])
            let original = try NativePNG.decode(Data(contentsOf: URL(fileURLWithPath: try XCTUnwrap(source["path"] as? String))))
            XCTAssertEqual(try NativePNG.decode(Data(contentsOf: sample.targetURL)).pixels, try original.crop([128, 128, 256, 256]).pixels)
        }
        _ = try await output(["cleanup-size", "--dataset", parallelRoot.path])
    }

    func testCancelledParallelPreparationJoinsWritersAndRemovesOnlyItsStage() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await preparationFixture(root, materials: 4, dimension: 1024)
        let original = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let policy = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: MachineResources.gibibyte, maximumWorkers: 4)
        let operation = Task {
            try await NativeMaterialDatasetService.run(arguments: ["prepare-size", "--dataset", dataset.path, "--size", "512", "--target", "height"], preparationPolicy: policy)
        }
        let staging = dataset.appendingPathComponent(".training-data")
        var found = false
        for _ in 0..<400 {
            if let names = try? FileManager.default.contentsOfDirectory(atPath: staging.path), names.contains(where: {
                $0.hasPrefix(".preparing-") && ((try? FileManager.default.contentsOfDirectory(atPath: staging.appendingPathComponent($0 + "/samples").path).isEmpty) == false)
            }) { found = true; break }
            try await Task.sleep(for: .milliseconds(2))
        }
        XCTAssertTrue(found, "Cancellation exercises live crop writers")
        operation.cancel()
        do { _ = try await operation.value; XCTFail("Cancellation must prevent publication") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: staging.path), [])
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), original)
        // Once the cancelled worker join returns, no background writer can
        // recreate files or keep the source dataset locked.
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--description", "After cancellation"])
        try await Task.sleep(for: .milliseconds(30))
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: staging.path), [])
    }

    func testFailedParallelDecodePreservesSourcesAndNeverPublishesPartialDataset() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = try await preparationFixture(root, materials: 3, dimension: 512)
        let index = try object(dataset.appendingPathComponent("dataset.json"))
        let first = try XCTUnwrap((index["samples"] as? [[String: Any]])?.first)
        let sampleURL = dataset.appendingPathComponent(try XCTUnwrap(first["path"] as? String)).appendingPathComponent("sample.json")
        var sample = try object(sampleURL), maps = try XCTUnwrap(sample["map_metadata"] as? [String: [String: Any]])
        var details = try XCTUnwrap(maps["height"]), source = try XCTUnwrap(details["source"] as? [String: Any])
        let original = URL(fileURLWithPath: try XCTUnwrap(source["path"] as? String))
        // Valid, checksum-bound IHDR reaches the worker; missing IDAT must
        // fail decoding while sibling materials may already be writing crops.
        let truncated = try Data(contentsOf: original).prefix(33)
        try truncated.write(to: original)
        let digest = SHA256.hash(data: truncated).map { String(format: "%02x", $0) }.joined()
        source["file_sha256"] = digest; source["file_bytes"] = truncated.count
        details["source"] = source; details["sample_sha256"] = digest; maps["height"] = details; sample["map_metadata"] = maps
        try JSONSerialization.data(withJSONObject: sample, options: [.sortedKeys]).write(to: sampleURL)
        let indexBytes = try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), metadataBytes = try Data(contentsOf: sampleURL)
        let policy = NativeMaterialDatasetService.PreparationPolicy(memoryBudgetBytes: MachineResources.gibibyte, maximumWorkers: 3)
        do {
            _ = try await preparationOutput(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height"], policy: policy)
            XCTFail("An incomplete source cannot publish generated samples")
        } catch { XCTAssertFalse(error is CancellationError) }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: dataset.appendingPathComponent(".training-data").path), [])
        XCTAssertEqual(try Data(contentsOf: original), truncated)
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), indexBytes)
        XCTAssertEqual(try Data(contentsOf: sampleURL), metadataBytes)
        _ = try await output(["edit-dataset", "--dataset", dataset.path, "--description", "After failed workers"])
    }

    private func preparationOutput(_ arguments: [String], policy: NativeMaterialDatasetService.PreparationPolicy, events: PreparationEventRecorder? = nil) async throws -> [String: Any] {
        let returned = try await NativeMaterialDatasetService.run(arguments: arguments, preparationPolicy: policy, onEvent: { events?.append($0) })
        let text = try XCTUnwrap(returned)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }
    private func preparationFixture(_ root: URL, materials: Int, dimension: Int) async throws -> URL {
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Parallel exact maps", "--training-size", "256"])
        for number in 0..<materials {
            let input = root.appendingPathComponent("input-\(number).png"), height = root.appendingPathComponent("height-\(number).png")
            var seed = UInt64(number + 1)
            func pixels(_ count: Int) -> Data {
                var result = Data(count: count)
                result.withUnsafeMutableBytes { bytes in
                    let values = bytes.bindMemory(to: UInt8.self)
                    for index in 0..<count { seed = seed &* 6364136223846793005 &+ 1442695040888963407; values[index] = UInt8(truncatingIfNeeded: seed >> 32) }
                }
                return result
            }
            try NativePNG(header: .init(width: dimension, height: dimension, bits: 8, channels: 3, color: 2, interlace: 0), pixels: pixels(dimension * dimension * 3), colorChunks: []).encoded().write(to: input)
            try NativePNG(header: .init(width: dimension, height: dimension, bits: 16, channels: 1, color: 0, interlace: 0), pixels: pixels(dimension * dimension * 2), colorChunks: []).encoded().write(to: height)
            _ = try await output(["add-material", "--dataset", dataset.path, "--name", "Material \(number)", "--input", input.path, "--height", height.path])
        }
        return dataset
    }
    private func preparedInventory(_ root: URL) throws -> [String: Data] {
        var result: [String: Data] = [:]
        for case let file as URL in FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey])! {
            if try file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true, file.lastPathComponent != "dataset.json", file.lastPathComponent != ".material-workbench.lock" {
                let path = file.resolvingSymlinksInPath().standardizedFileURL.path, prefix = root.resolvingSymlinksInPath().standardizedFileURL.path + "/"
                if file.lastPathComponent == "sample.json" {
                    // Generated file timestamps vary across otherwise identical
                    // preparations; compare source bindings and crop metadata.
                    func stable(_ value: Any) -> Any {
                        if let object = value as? [String: Any] { return object.filter { $0.key != "prepared_stat" }.mapValues(stable) }
                        if let array = value as? [Any] { return array.map(stable) }
                        return value
                    }
                    result[String(path.dropFirst(prefix.count))] = try JSONSerialization.data(withJSONObject: stable(object(file)), options: [.sortedKeys])
                } else { result[String(path.dropFirst(prefix.count))] = try Data(contentsOf: file) }
            }
        }
        return result
    }

    func testPNGPreservesNativeChannelsAndCodesThroughExactCrop() throws {
        for bits in [8, 16] { for (color, channels) in [(UInt8(0), 1), (UInt8(2), 3), (UInt8(4), 2), (UInt8(6), 4)] {
            let pixels = Data((0..<3 * 2 * channels * bits / 8).map { UInt8(truncatingIfNeeded: $0 * 97) })
            let source = NativePNG(header: .init(width: 3, height: 2, bits: bits, channels: channels, color: color, interlace: 0), pixels: pixels, colorChunks: [("gAMA", Data([0, 0, 177, 143]))])
            let decoded = try NativePNG.decode(source.encoded())
            XCTAssertEqual(decoded.header, source.header); XCTAssertEqual(decoded.pixels, pixels)
            XCTAssertEqual(decoded.colorChunks.first?.1, source.colorChunks.first?.1)
            let selected = try decoded.crop([1, 1, 2, 1])
            XCTAssertEqual(selected.pixels, pixels.subdata(in: 4 * source.header.bytesPerPixel..<6 * source.header.bytesPerPixel))
            XCTAssertEqual(try NativePNG.decode(selected.encoded()).pixels, selected.pixels)
        } }
    }
    func testPNGNormalConventionChangesOnlyGreenIntegerCodes() throws {
        let source = NativePNG(header: .init(width: 2, height: 1, bits: 16, channels: 4, color: 6, interlace: 0), pixels: Data([0, 1, 0, 0, 255, 255, 0, 19, 1, 0, 128, 1, 0, 0, 2, 0]), colorChunks: [])
        let result = try source.crop([0, 0, 2, 1], flipGreen: true)
        XCTAssertEqual(result.pixels, Data([0, 1, 255, 255, 255, 255, 0, 19, 1, 0, 127, 254, 0, 0, 2, 0]))
        XCTAssertEqual(try result.crop([0, 0, 2, 1], flipGreen: true).pixels, source.pixels)
        XCTAssertThrowsError(try source.crop([-1, 0, 2, 1]))
    }
    func testIndependentPNGFixturesDecodeAllFiltersAndAdam7WithoutChangingCodes() throws {
        // Independent zlib fixtures cover filter bytes 0...4 and all seven
        // Adam7 passes. Values are deliberately asymmetric and span both bytes.
        let filtered = try XCTUnwrap(Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAMAAAAFEAIAAABfgx22AAAARElEQVR4nGNgSDyk3PLUbTlH5intnrd+6wUKGS8ZT/katt0NCTBdwgDMKldZzZ6J8/Tw9AARD4hmgchIXJIAoktuIBoASx8sjjtVTA0AAAAASUVORK5CYII="))
        let adam7 = try XCTUnwrap(Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAEEAAAAAFEz0ZJAAAAO0lEQVR4nAEwAM//AABhAAhpAITlAJT1GHmc/QDCI0anAFa32jsAyiuM7U6vEHHSMwBevyCB4kOkBWbHt1wTjZvigdQAAAAASUVORK5CYII="))
        XCTAssertEqual(try NativePNG.decode(filtered).pixels, Data((0..<90).map { UInt8(truncatingIfNeeded: $0 * 97) }))
        XCTAssertEqual(try NativePNG.decode(adam7).pixels, Data((0..<40).map { UInt8(truncatingIfNeeded: $0 * 97) }))
    }
    func testNativePNGAcceptsSupportedTransparencyAndRejectsAmbiguousHeaders() throws {
        let source = NativePNG(header: .init(width: 2, height: 1, bits: 8, channels: 3, color: 2, interlace: 0), pixels: Data(repeating: 0, count: 6), colorChunks: [("tRNS", Data(repeating: 0, count: 6))])
        let decoded = try NativePNG.decode(source.encoded())
        XCTAssertEqual(decoded.pixels, source.pixels)
        XCTAssertEqual(decoded.colorChunks.first?.1, source.colorChunks.first?.1)
        XCTAssertEqual(try decoded.modelFloatSamples(role: "input", encoding: "linear_rgb"), [Float](repeating: 0, count: 6))
        let malformed = NativePNG(header: source.header, pixels: source.pixels, colorChunks: [("tRNS", Data(repeating: 0, count: 2))])
        XCTAssertThrowsError(try malformed.encoded())
        let plain = NativePNG(header: source.header, pixels: source.pixels, colorChunks: [])
        var duplicated = try plain.encoded(); duplicated.insert(contentsOf: duplicated.subdata(in: 8..<33), at: 33)
        XCTAssertThrowsError(try NativePNG.decode(duplicated))
    }
    func testPNGRejectsChecksumChangesAndIncompleteStreams() throws {
        let source = NativePNG(header: .init(width: 2, height: 1, bits: 8, channels: 1, color: 0, interlace: 0), pixels: Data([0, 255]), colorChunks: [])
        var bytes = try source.encoded(); bytes[29] ^= 1
        XCTAssertThrowsError(try NativePNG.decode(bytes))
        XCTAssertThrowsError(try NativePNG.decode(source.encoded().dropLast()))
    }
    func testNativeCreateEditConflictAndOriginalPreservation() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Native Dataset")
        let created = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Native Dataset", "--training-size", "256"])
        XCTAssertEqual(created["material_count"] as? Int, 0)
        let hash = try XCTUnwrap(created["index_sha256"] as? String)
        let edited = try await output(["edit-dataset", "--dataset", dataset.path, "--name", "Renamed", "--description", "Exact originals", "--expected-index-sha256", hash])
        XCTAssertEqual(edited["name"] as? String, "Renamed")
        do { _ = try await output(["edit-dataset", "--dataset", dataset.path, "--validation-subject", "missing", "--subject-validation", "automatic"]); XCTFail("A stale subject cannot be reset") }
        catch { XCTAssertTrue(error.localizedDescription.contains("no longer in this dataset")) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: dataset.appendingPathComponent(".material-workbench-journal.json").path))
        do { _ = try await output(["edit-dataset", "--dataset", dataset.path, "--name", "Stale", "--expected-index-sha256", hash]); XCTFail("A stale index must not edit metadata") }
        catch { XCTAssertTrue(error.localizedDescription.contains("changed since selection")) }
        let bytes = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let delete = try await output(["validate-delete", "--dataset", dataset.path, "--expected-index-sha256", try XCTUnwrap(edited["index_sha256"] as? String)])
        XCTAssertEqual(delete["safe_to_trash_folder"] as? Bool, true)
        try Data([1, 2, 3]).write(to: dataset.appendingPathComponent("original.png"))
        let conservative = try await output(["validate-delete", "--dataset", dataset.path])
        XCTAssertEqual(conservative["safe_to_trash_folder"] as? Bool, false)
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), bytes)
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("original.png")), Data([1, 2, 3]))
    }
    func testValidationCropCountsPersistZeroAndValuesBeyondFormerUICaps() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        let created = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Validation preferences",
            "--validation-enabled", "no", "--validation-percent", "0", "--validation-max-crops", "0", "--validation-quick-count", "0"])
        let disabled = try XCTUnwrap(created["validation"] as? [String: Any])
        XCTAssertEqual(disabled["enabled"] as? Bool, false)
        XCTAssertEqual(disabled["percent"] as? Double, 0)
        XCTAssertEqual(disabled["max_crops"] as? Int, 0)
        XCTAssertEqual(disabled["quick_count"] as? Int, 0)

        for count in [100_001, Int.max, 0] {
            _ = try await output(["edit-dataset", "--dataset", dataset.path, "--validation-enabled", "yes",
                "--validation-max-crops", String(count), "--validation-quick-count", String(count)])
            let reopened = try await output(["dataset", "--dataset", dataset.path])
            let settings = try XCTUnwrap(reopened["validation"] as? [String: Any])
            XCTAssertEqual(settings["max_crops"] as? Int, count)
            XCTAssertEqual(settings["quick_count"] as? Int, count)
        }
        let acceptedBytes = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        for field in ["--validation-max-crops", "--validation-quick-count"] {
            do {
                _ = try await output(["edit-dataset", "--dataset", dataset.path, field, "-1"])
                XCTFail("Negative crop counts are outside the action's range.")
            } catch {
                XCTAssertTrue(error.localizedDescription.contains("nonnegative crop counts"))
            }
            XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), acceptedBytes)
        }
    }

    func testNativeReadReconstructsSharedCropPlansAndRejectsPathEscape() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Dataset", "--training-size", "256", "--validation-percent", "25"])
        var index = try object(dataset.appendingPathComponent("dataset.json"))
        var entries: [[String: Any]] = [], sourceBytes: [URL: Data] = [:]
        for number in 0..<4 {
            let material = "material_\(number)", id = material + "_full"
            let folder = root.appendingPathComponent("source_\(number)")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
            var maps: [String: String] = [:], metadata: [String: Any] = [:]
            for (role, bits) in [("input", 8), ("height", 16)] {
                let file = folder.appendingPathComponent(role + ".png")
                let png = NativePNG(header: .init(width: 512, height: 512, bits: bits, channels: 1, color: 0, interlace: 0), pixels: Data(repeating: UInt8(number * 31), count: 512 * 512 * bits / 8), colorChunks: [])
                let data = try png.encoded(); try data.write(to: file); sourceBytes[file] = data
                let hash = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                maps[role] = file.path
                metadata[role] = ["storage": "source_reference", "sample_sha256": hash, "sample_bits": bits, "source": ["path": file.path, "width": 512, "height": 512, "sample_bits": bits, "file_sha256": hash]]
            }
            let entry: [String: Any] = ["sample_id": id, "material_id": material, "status": "approved", "split": "train", "path": "samples/" + id]
            entries.append(entry)
            let sample: [String: Any] = ["sample_id": id, "material_id": material, "status": "approved", "split": "train", "source_directory": folder.path, "source_pixel_dimensions": [512, 512], "sample_pixel_dimensions": [512, 512], "maps": maps, "map_metadata": metadata]
            let metadataFolder = dataset.appendingPathComponent("samples/" + id)
            try FileManager.default.createDirectory(at: metadataFolder, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: sample).write(to: metadataFolder.appendingPathComponent("sample.json"))
        }
        index["samples"] = entries
        try JSONSerialization.data(withJSONObject: index).write(to: dataset.appendingPathComponent("dataset.json"))
        let inspected = try await output(["dataset", "--dataset", dataset.path, "--target", "height"])
        XCTAssertEqual(inspected["material_count"] as? Int, 4); XCTAssertEqual(inspected["sample_count"] as? Int, 5)
        let plans = try XCTUnwrap(inspected["training_plans"] as? [String: [String: Any]])
        XCTAssertEqual(plans["height"]?["train_count"] as? Int, 4)
        XCTAssertEqual(plans["height"]?["validation_count"] as? Int, 1)
        XCTAssertEqual(plans["roughness"]?["unavailable_target_count"] as? Int, 5)
        for (file, bytes) in sourceBytes { XCTAssertEqual(try Data(contentsOf: file), bytes) }
        entries[0]["path"] = "../escaped"; index["samples"] = entries
        try JSONSerialization.data(withJSONObject: index).write(to: dataset.appendingPathComponent("dataset.json"))
        do { _ = try await output(["dataset", "--dataset", dataset.path]); XCTFail("Metadata path escapes must fail") }
        catch { XCTAssertTrue(error.localizedDescription.contains("inside its directory")) }
    }
    func testCancellationBeforeNativeCommandDoesNotCreateDataset() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Cancelled")
        let task = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await NativeMaterialDatasetService.run(arguments: ["create-dataset", "--dataset", dataset.path, "--name", "Cancelled"])
        }
        do { _ = try await task.value; XCTFail("Cancelled command should fail") } catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: dataset.path))
    }
    func testCurrentSourceInventoryRefreshIsStableAcrossRepeatedReads() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("Dataset")
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Native", "--training-size", "256"])
        let folder = dataset.appendingPathComponent("sources/Brick")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let input = folder.appendingPathComponent("brick_diff_1k.png"), height = folder.appendingPathComponent("brick_disp_1k.png")
        let rgb = NativePNG(header: .init(width: 256, height: 256, bits: 8, channels: 3, color: 2, interlace: 0), pixels: Data(repeating: 127, count: 256 * 256 * 3), colorChunks: [])
        let numeric = NativePNG(header: .init(width: 256, height: 256, bits: 16, channels: 1, color: 0, interlace: 0), pixels: Data(repeating: 97, count: 256 * 256 * 2), colorChunks: [])
        let inputBytes = try rgb.encoded(), heightBytes = try numeric.encoded()
        try inputBytes.write(to: input); try heightBytes.write(to: height)
        let first = try await output(["dataset", "--dataset", dataset.path])
        XCTAssertEqual(first["material_count"] as? Int, 1)
        let indexURL = dataset.appendingPathComponent("dataset.json"), metadata = try object(indexURL)
        XCTAssertEqual(metadata["source_discovery"] as? String, "registered-resolution-and-color-sets-v2")
        XCTAssertEqual((metadata["source_inventory_sha256"] as? String)?.count, 64)
        let indexBytes = try Data(contentsOf: indexURL)
        var before = stat(); XCTAssertEqual(stat(indexURL.path, &before), 0)
        let second = try await output(["dataset", "--dataset", dataset.path])
        var after = stat(); XCTAssertEqual(stat(indexURL.path, &after), 0)
        XCTAssertEqual(second["index_sha256"] as? String, first["index_sha256"] as? String)
        XCTAssertEqual(try Data(contentsOf: indexURL), indexBytes)
        XCTAssertEqual(before.st_mtimespec.tv_sec, after.st_mtimespec.tv_sec)
        XCTAssertEqual(before.st_mtimespec.tv_nsec, after.st_mtimespec.tv_nsec)
        XCTAssertEqual(before.st_ctimespec.tv_sec, after.st_ctimespec.tv_sec)
        XCTAssertEqual(before.st_ctimespec.tv_nsec, after.st_ctimespec.tv_nsec)
        XCTAssertEqual(try Data(contentsOf: input), inputBytes); XCTAssertEqual(try Data(contentsOf: height), heightBytes)
        var retired = metadata; retired["storage_policy"] = "copied-normalized-legacy"
        try JSONSerialization.data(withJSONObject: retired).write(to: indexURL)
        do { _ = try await output(["dataset", "--dataset", dataset.path]); XCTFail("Retired storage cannot be silently migrated") }
        catch { XCTAssertTrue(error.localizedDescription.contains("retired material schema")) }
        XCTAssertEqual(try Data(contentsOf: input), inputBytes); XCTAssertEqual(try Data(contentsOf: height), heightBytes)
    }
    func testNativeImportPrepareCleanupAndMissingSourceLifecycleKeepsOriginalCodes() async throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let sources = root.appendingPathComponent("Walls"), dataset = root.appendingPathComponent("Dataset")
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: false)
        let input = sources.appendingPathComponent("diffuse.png"), height = sources.appendingPathComponent("height.png")
        let rgb = NativePNG(header: .init(width: 512, height: 512, bits: 8, channels: 3, color: 2, interlace: 0), pixels: Data(repeating: 127, count: 512 * 512 * 3), colorChunks: [])
        let rawHeight = Data((0..<512 * 512 * 2).map { UInt8(truncatingIfNeeded: $0 * 97) })
        let numeric = NativePNG(header: .init(width: 512, height: 512, bits: 16, channels: 1, color: 0, interlace: 0), pixels: rawHeight, colorChunks: [])
        let inputBytes = try rgb.encoded(), heightBytes = try numeric.encoded()
        try inputBytes.write(to: input); try heightBytes.write(to: height)
        _ = try await output(["create-dataset", "--dataset", dataset.path, "--name", "Native", "--training-size", "256"])
        let planURL = root.appendingPathComponent("preview.json")
        let preview = try await output(["scan-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path])
        XCTAssertEqual(preview["added_material_count"] as? Int, 1)
        let imported = try await output(["import-folder", "--dataset", dataset.path, "--folder", sources.path, "--plan", planURL.path, "--expected-plan-sha256", try XCTUnwrap(preview["plan_sha256"] as? String), "--expected-index-sha256", try XCTUnwrap(preview["index_sha256"] as? String), "--training-size", "256"])
        let originalIndex = try Data(contentsOf: dataset.appendingPathComponent("dataset.json"))
        let materials = try XCTUnwrap(imported["materials"] as? [[String: Any]])
        let sample = try XCTUnwrap((materials[0]["samples"] as? [[String: Any]])?.first)
        let sampleID = try XCTUnwrap(sample["sample_id"] as? String)
        _ = try await output(["curate", "--dataset", dataset.path, "--sample", sampleID, "--review-size", "256", "--status", "approved", "--note", "Keep relief", "--expected-review-sha256", try XCTUnwrap(imported["review_sha256"] as? String)])
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), originalIndex, "Crop reviews only update their source-bound sidecar")
        let prepared = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height", "--automatic-validation"])
        let preparedPath = try XCTUnwrap(prepared["dataset_path"] as? String), preparedURL = URL(fileURLWithPath: preparedPath)
        let descriptors = try NativeMaterialDatasetService.trainingSamples(datasetURL: preparedURL, size: 256, target: "height")
        XCTAssertEqual(descriptors.count, 1)
        let descriptor = try XCTUnwrap(descriptors.first)
        let preparedHeight = try NativePNG.decode(Data(contentsOf: descriptor.targetURL))
        XCTAssertEqual(preparedHeight.pixels, try numeric.crop([128, 128, 256, 256]).pixels)
        XCTAssertEqual(try preparedHeight.modelFloatSamples(role: "height").count, 256 * 256)
        let cropBytes = try Data(contentsOf: descriptor.targetURL)
        let reviewBytes = try Data(contentsOf: dataset.appendingPathComponent(".material-size-reviews.json"))
        try Data([1, 2, 3]).write(to: descriptor.targetURL)
        do { _ = try await output(["cleanup-size", "--dataset", preparedPath]); XCTFail("Modified crops need inspection before cleanup") }
        catch { XCTAssertTrue(error.localizedDescription.contains("crop contents changed")) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: preparedPath))
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent(".material-size-reviews.json")), reviewBytes)
        try cropBytes.write(to: descriptor.targetURL)
        let unrelated = preparedURL.appendingPathComponent("Unrelated")
        try FileManager.default.createDirectory(at: unrelated, withIntermediateDirectories: false)
        do { _ = try await output(["cleanup-size", "--dataset", preparedPath]); XCTFail("Empty unrelated folders cannot be removed with crops") }
        catch { XCTAssertTrue(error.localizedDescription.contains("unrelated folder")) }
        try FileManager.default.removeItem(at: unrelated)
        let pipe = preparedURL.appendingPathComponent("unrelated-pipe")
        XCTAssertEqual(mkfifo(pipe.path, S_IRUSR | S_IWUSR), 0)
        do { _ = try await output(["cleanup-size", "--dataset", preparedPath]); XCTFail("Special files cannot be removed with crops") }
        catch { XCTAssertTrue(error.localizedDescription.contains("special file")) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: pipe.path))
        try FileManager.default.removeItem(at: pipe)
        let reused = try await output(["prepare-size", "--dataset", dataset.path, "--size", "256", "--target", "height", "--automatic-validation"])
        XCTAssertEqual((reused["preparation"] as? [String: Any])?["reused"] as? Bool, true)
        _ = try await output(["cleanup-size", "--dataset", preparedPath])
        XCTAssertFalse(FileManager.default.fileExists(atPath: preparedPath))
        XCTAssertEqual(try Data(contentsOf: input), inputBytes); XCTAssertEqual(try Data(contentsOf: height), heightBytes)
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), originalIndex)
        let inspected = try await output(["dataset", "--dataset", dataset.path])
        let inspectedMaterials = inspected["materials"] as? [[String: Any]]
        let inspectedSamples = inspectedMaterials?.first?["samples"] as? [[String: Any]]
        XCTAssertEqual(inspectedSamples?.first?["note"] as? String, "Keep relief")
        _ = try await output(["remove-missing", "--dataset", dataset.path, "--sample", sampleID, "--review-size", "256", "--path", height.path])
        XCTAssertEqual(try Data(contentsOf: dataset.appendingPathComponent("dataset.json")), originalIndex, "Present originals cannot be removed by a missing-source operation")
        try FileManager.default.removeItem(at: height)
        let removed = try await output(["remove-missing", "--dataset", dataset.path, "--sample", sampleID, "--review-size", "256", "--path", height.resolvingSymlinksInPath().path])
        XCTAssertEqual(removed["removed"] as? Bool, true)
        XCTAssertEqual(try Data(contentsOf: input), inputBytes)
    }
    func testModelBoundaryUsesPlanarFloatCodesAndKeepsIntegerSourceUntouched() throws {
        let codes = Data([0, 128, 255, 255, 64, 192, 0, 255])
        let png = NativePNG(header: .init(width: 2, height: 1, bits: 8, channels: 4, color: 6, interlace: 0), pixels: codes, colorChunks: [])
        XCTAssertEqual(try png.modelFloatSamples(role: "input", encoding: "source_srgb_assumed"), [0, Float(64) / 255, Float(128) / 255, Float(192) / 255, 1, 0])
        let normal = try png.modelFloatSamples(role: "normal", normalConvention: "DirectX")
        XCTAssertEqual(normal[2], Float(127) / 255); XCTAssertEqual(normal[3], Float(63) / 255)
        XCTAssertEqual(png.pixels, codes)
        XCTAssertThrowsError(try png.modelFloatSamples(role: "height"))
        XCTAssertThrowsError(try png.modelFloatSamples(role: "input", encoding: "unknown"))
    }
    private func fixturePNG(width: Int, height: Int, bits: Int, channels: Int) throws -> Data {
        let pixels = Data((0..<width * height * channels * bits / 8).map { UInt8(truncatingIfNeeded: $0 * 97) })
        return try NativePNG(header: .init(width: width, height: height, bits: bits, channels: channels,
                                          color: channels == 1 ? 0 : 2, interlace: 0), pixels: pixels, colorChunks: []).encoded()
    }
    private func output(_ arguments: [String]) async throws -> [String: Any] {
        let returned = try await NativeMaterialDatasetService.run(arguments: arguments)
        let text = try XCTUnwrap(returned)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
    }
    private func object(_ url: URL) throws -> [String: Any] { try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]) }
    private func temporary() throws -> URL { let root = FileManager.default.temporaryDirectory.appendingPathComponent("native-material-\(UUID().uuidString)"); try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false); return root }
}

private final class PreparationEventRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [String] = []
    func append(_ value: String) { lock.withLock { values.append(value) } }
    func objects() throws -> [[String: Any]] {
        try lock.withLock { try values.map { try JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] } }
    }
}

private final class SourceScanProbe: @unchecked Sendable {
    private let condition = NSCondition()
    private var active = 0, peak = 0, total = 0
    func begin(requireConcurrentReaders: Int) throws {
        condition.lock(); defer { condition.unlock() }
        active += 1; total += 1; peak = max(peak, active); condition.broadcast()
        let deadline = Date().addingTimeInterval(5)
        while total < requireConcurrentReaders {
            if !condition.wait(until: deadline) { active -= 1; throw StudioError("Source scan did not inspect files concurrently.") }
        }
    }
    func end() { condition.lock(); active -= 1; condition.broadcast(); condition.unlock() }
    func waitForReaders(_ minimum: Int) -> Bool {
        condition.lock(); defer { condition.unlock() }
        let deadline = Date().addingTimeInterval(5)
        while total < minimum { if !condition.wait(until: deadline) { return false } }
        return true
    }
    var activeReaders: Int { condition.lock(); defer { condition.unlock() }; return active }
    var peakReaders: Int { condition.lock(); defer { condition.unlock() }; return peak }
    var totalReaders: Int { condition.lock(); defer { condition.unlock() }; return total }
}

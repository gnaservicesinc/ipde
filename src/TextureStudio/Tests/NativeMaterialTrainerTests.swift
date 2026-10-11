import Darwin
import Foundation
import Metal
import XCTest
@testable import TextureStudio

final class NativeMaterialTrainerTests: XCTestCase {
    func testCadenceUsesEpochByDefaultAndRejectsRetiredOrInvalidUnits() throws {
        let arguments = ["train", "--dataset", "/dataset", "--output", "/output"]
        let options = try NativeMaterialTrainer.Options(arguments)
        XCTAssertEqual(options.validationEvery, 0)
        XCTAssertEqual(options.validationUnit, .epoch)
        XCTAssertEqual(options.checkpointUnit, .epoch)
        for flag in ["--validation-unit", "--checkpoint-unit"] {
            for unit in ["updates", "invalid", ""] {
                XCTAssertThrowsError(try NativeMaterialTrainer.Options(arguments + [flag, unit]))
            }
        }
        XCTAssertFalse(NativeMaterialTrainer.cadenceDue(every: 2, unit: .epoch, steps: 8, epochs: 1, epochBoundary: false))
        XCTAssertTrue(NativeMaterialTrainer.cadenceDue(every: 2, unit: .epoch, steps: 8, epochs: 2, epochBoundary: true))
        XCTAssertTrue(NativeMaterialTrainer.cadenceDue(every: 2, unit: .step, steps: 8, epochs: 1, epochBoundary: false))
        XCTAssertFalse(NativeMaterialTrainer.cadenceDue(every: 2, unit: .step, steps: 8, epochs: 2, epochBoundary: true))
    }

    func testEpochAndStepCadenceTriggerAtDifferentSuccessfulBoundaries() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset")
        try datasetFixture(dataset, trainingCount: 2)
        for (validationUnit, checkpointUnit, expectedSaves, expectedQuick) in [
            ("epoch", "epoch", [4, 6], [2, 6]), ("step", "step", [2, 4, 6], [1, 3, 5]),
            ("step", "epoch", [4, 6], [1, 2, 3, 5, 6]), ("epoch", "step", [2, 4, 6], [])
        ] {
            let output = root.appendingPathComponent(validationUnit + "-" + checkpointUnit), model = try trainerModel(), events = Recorder()
            let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
                "--size", "256", "--updates-per-map", "3", "--validation-every", "1", "--validation-unit", validationUnit,
                "--checkpoint-every", "2", "--checkpoint-unit", checkpointUnit])
            _ = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: events.append, control: .init(), model: model) }.value
            let saves = events.events.filter { $0["event"] as? String == "checkpoint_saved" }.compactMap { $0["completed_updates"] as? Int }
            let quick = events.events.filter { $0["event"] as? String == "validation" && $0["context"] as? String == "periodic" }.compactMap { $0["completed_updates"] as? Int }
            XCTAssertEqual(saves, expectedSaves)
            XCTAssertEqual(quick, expectedQuick)
        }
    }

    func testMalformedAndChangedTrainingMapsAreQuarantinedAcrossEpochs() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset"), output = root.appendingPathComponent("trained")
        try datasetFixture(dataset, validationEnabled: false, trainingCount: 3)
        try Data("changed".utf8).write(to: dataset.appendingPathComponent("samples/train-0/diffuse.png"))
        let malformed = Data("malformed png".utf8), sampleURL = dataset.appendingPathComponent("samples/train-1/sample.json")
        try malformed.write(to: dataset.appendingPathComponent("samples/train-1/diffuse.png"))
        var sample = try NativeMaterialTransfer.object(sampleURL)
        var metadata = try XCTUnwrap(sample["map_metadata"] as? [String: [String: Any]])
        metadata["input"]?["sample_sha256"] = NativeMaterialTrainer.checksum(malformed)
        sample["map_metadata"] = metadata
        try JSONSerialization.data(withJSONObject: sample).write(to: sampleURL)
        let model = try trainerModel(), events = Recorder()
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
            "--size", "256", "--updates-per-map", "2", "--checkpoint-every", "1"])
        let text = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: events.append, control: .init(), model: model) }.value
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        XCTAssertEqual(result["status"] as? String, "completed")
        XCTAssertEqual(result["completed_updates"] as? Int, 2)
        XCTAssertEqual(result["completed_epochs"] as? Int, 2)
        XCTAssertEqual(result["skipped_sample_count"] as? Int, 2)
        XCTAssertEqual(events.events.filter { $0["event"] as? String == "sample_skipped" }.count, 2)
        XCTAssertEqual(events.events.filter { $0["event"] as? String == "update" }.compactMap { $0["sample_id"] as? String }, ["train-2", "train-2"])
        XCTAssertNoThrow(try NativeMaterialPackage.verify(output.appendingPathComponent("export")))
        XCTAssertTrue(FileManager.default.fileExists(atPath: output.appendingPathComponent("skipped-samples.json").path))
    }

    func testFinalValidationFailureStillPublishesTrainedCheckpointAndPackage() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset"), output = root.appendingPathComponent("trained")
        try datasetFixture(dataset)
        let model = try trainerModel(), events = Recorder(), original = model.adapterWeights
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
            "--size", "256", "--updates-per-map", "2", "--validation-every", "0"])
        let text = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: { line in
            events.append(line)
            if events.updates == 2 { try? FileManager.default.removeItem(at: dataset.appendingPathComponent("samples/validation/height.png")) }
        }, control: .init(), model: model) }.value
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        XCTAssertEqual(result["status"] as? String, "completed")
        XCTAssertEqual(result["completed_updates"] as? Int, 2)
        let validation = try XCTUnwrap(result["final_validation"] as? [String: Any])
        XCTAssertEqual(validation["status"] as? String, "unavailable")
        XCTAssertEqual(validation["sample_count"] as? Int, 0)
        XCTAssertEqual(validation["validation_skipped_sample_count"] as? Int, 1)
        XCTAssertTrue(validation["mae"] is NSNull)
        let saved = try NativeSafetensors(contentsOf: output.appendingPathComponent("checkpoint-step-00000002.safetensors"))
        XCTAssertNotEqual(try saved.tensorBytes(named: "ups.3.model.10.lora_B"), original["ups.3.model.10.lora_B"]!.bytes)
        XCTAssertNoThrow(try NativeMaterialPackage.verify(output.appendingPathComponent("export")))
    }

    func testSkippedMicrobatchesFlushOnlyValidGradientsAndAllBadSamplesSaveProgress() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset"), output = root.appendingPathComponent("trained")
        try datasetFixture(dataset, validationEnabled: false)
        let model = try trainerModel(), events = Recorder()
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
            "--size", "256", "--updates-per-map", "3", "--gradient-accumulation-steps", "3"])
        let text = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: { line in
            events.append(line)
            if let event = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
               event["event"] as? String == "accumulation_sample", event["accumulation_step"] as? Int == 2 {
                try? FileManager.default.removeItem(at: dataset.appendingPathComponent("samples/train/diffuse.png"))
            }
        }, control: .init(), model: model) }.value
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        XCTAssertEqual(result["status"] as? String, "stopped")
        XCTAssertEqual(result["stopped_reason"] as? String, "no_valid_training_samples")
        XCTAssertEqual(result["completed_updates"] as? Int, 1)
        XCTAssertEqual(result["sample_evaluations"] as? Int, 1)
        XCTAssertEqual(result["skipped_sample_count"] as? Int, 1)
        XCTAssertEqual(events.events.first { $0["event"] as? String == "update" }?["accumulated_samples"] as? Int, 1)
        XCTAssertNoThrow(try NativeMaterialPackage.verify(output.appendingPathComponent("export")))
    }

    func testAccumulationKeepsOptimizerStepPlanAndRecordsEffectiveCheckpointConfiguration() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset")
        try datasetFixture(dataset, validationEnabled: false)
        var finalFactors: [[String: NativeTensor]] = []
        for accumulation in [1, 3] {
            let model = try trainerModel(), events = Recorder()
            let output = root.appendingPathComponent("trained-\(accumulation)")
            let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
                "--size", "256", "--updates-per-map", "2", "--gradient-accumulation-steps", String(accumulation),
                "--optimizer", "adamw", "--learning-rate", "0.001", "--weight-decay", "0.01",
                "--optimizer-beta1", "0.8", "--optimizer-beta2", "0.99", "--optimizer-epsilon", "0.0000001",
                "--max-gradient-norm", "0.5", "--seed", "19"])
            let text = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: events.append, control: .init(), model: model) }.value
            let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
            XCTAssertEqual(result["requested_updates"] as? Int, 2)
            XCTAssertEqual(result["completed_updates"] as? Int, 2)
            XCTAssertEqual(result["sample_evaluations"] as? Int, 2 * accumulation)
            let updates = events.events.filter { $0["event"] as? String == "update" }
            XCTAssertEqual(updates.count, 2)
            XCTAssertTrue(updates.allSatisfy { $0["accumulated_samples"] as? Int == accumulation })
            let configuration = try NativeMaterialTransfer.object(output.appendingPathComponent("export/config.json"))
            let effective = try XCTUnwrap(configuration["training_configuration"] as? [String: Any])
            XCTAssertEqual(effective["optimizer"] as? String, "adamw")
            XCTAssertEqual(effective["seed"] as? Int, 19)
            XCTAssertEqual(effective["gradient_accumulation_steps"] as? Int, accumulation)
            XCTAssertEqual(effective["optimizer_beta1"] as? Double ?? 0, 0.8, accuracy: 1e-6)
            XCTAssertEqual(effective["weight_decay"] as? Double ?? 0, 0.01, accuracy: 1e-6)
            XCTAssertEqual(effective["optimizer_state_restored"] as? Bool, false)
            let checkpoint = try NativeSafetensors(contentsOf: output.appendingPathComponent("checkpoint-step-00000002.safetensors"))
            let checkpointConfiguration = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(try XCTUnwrap(checkpoint.metadata["configuration"]).utf8)) as? [String: Any])
            XCTAssertNotNil(checkpointConfiguration["training_configuration"])
            finalFactors.append(model.adapterWeights)
        }
        // Three copies of the same pair have the same averaged derivative as
        // one copy, with weights frozen throughout each accumulation group.
        for name in finalFactors[0].keys {
            for (single, accumulated) in zip(try finalFactors[0][name]!.floatValues(), try finalFactors[1][name]!.floatValues()) {
                XCTAssertEqual(single, accumulated, accuracy: 1e-6)
            }
        }
    }

    func testStopAndSaveFlushesPartialAccumulationUsingActualMapCount() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset"), output = root.appendingPathComponent("trained")
        try datasetFixture(dataset, validationEnabled: false)
        let model = try trainerModel(), events = Recorder(), control = NativeMaterialTrainingControl()
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
            "--size", "256", "--updates-per-map", "3", "--gradient-accumulation-steps", "4"])
        let text = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: { line in
            events.append(line)
            if let event = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
               event["event"] as? String == "accumulation_sample", event["accumulation_step"] as? Int == 2 { control.stopAndSave() }
        }, control: control, model: model) }.value
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        XCTAssertEqual(result["status"] as? String, "stopped")
        XCTAssertEqual(result["completed_updates"] as? Int, 1)
        XCTAssertEqual(result["sample_evaluations"] as? Int, 2)
        XCTAssertEqual(events.events.first { $0["event"] as? String == "update" }?["accumulated_samples"] as? Int, 2)
        XCTAssertNoThrow(try NativeMaterialPackage.verify(output.appendingPathComponent("export")))
    }

    func testModelNameAcceptsDisplayTextAndRefinementDefaultsToRecordedName() throws {
        let arguments = ["train", "--dataset", "/dataset", "--output", "/output"]
        XCTAssertNil(try NativeMaterialTrainer.Options(arguments).modelName)
        XCTAssertEqual(try NativeMaterialTrainer.Options(arguments + ["--model-name", "  石 Stone / Warm evening  "]).modelName,
            "石 Stone / Warm evening")
        XCTAssertNil(try NativeMaterialTrainer.Options(arguments + ["--model-name", " \n "]).modelName)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let checkpoint = root.appendingPathComponent("adapter.safetensors")
        try NativeSafetensors.write(tensors: ["weight": .floats([1], shape: [1])],
            metadata: ["configuration": "{\"model_name\":\"Recorded stone\"}"], to: checkpoint)
        let refine = ["refine", "--dataset", "/dataset", "--output", "/output", "--checkpoint", checkpoint.path]
        XCTAssertEqual(try NativeMaterialTrainer.Options(refine).modelName, "Recorded stone")
        XCTAssertEqual(try NativeMaterialTrainer.Options(refine + ["--model-name", " \n "]).modelName, "Recorded stone")
        XCTAssertEqual(try NativeMaterialTrainer.Options(refine + ["--model-name", "New stone"]).modelName, "New stone")
    }

    func testQuickCheckCadenceAcceptsZeroAndHasNoArbitraryMaximum() throws {
        for value in [0, 1, Int.max] {
            let options = try NativeMaterialTrainer.Options(["train", "--dataset", "/dataset", "--output", "/output",
                "--validation-every", String(value)])
            XCTAssertEqual(options.validationEvery, value)
        }
        for value in ["-1", "invalid"] {
            XCTAssertThrowsError(try NativeMaterialTrainer.Options(["train", "--dataset", "/dataset", "--output", "/output",
                "--validation-every", value]))
        }
    }

    func testQuickChecksUseDatasetCropCountAndZeroDisablesThem() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        // A count larger than the old hard-coded three proves saved dataset
        // settings drive runtime selection. Both zero settings disable cadence.
        for (index, configuration) in [(0, (every: 0, quick: 4)), (1, (every: 1, quick: 0)), (2, (every: 1, quick: 4))] {
            let dataset = root.appendingPathComponent("dataset-\(index)")
            let output = root.appendingPathComponent("output-\(index)")
            try datasetFixture(dataset, validationCount: 5, quickCount: configuration.quick)
            let model = try trainerModel(), events = Recorder()
            let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
                "--size", "256", "--updates-per-map", "1", "--validation-every", String(configuration.every),
                "--max-minutes", String(Double.greatestFiniteMagnitude)])
            _ = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: events.append, control: .init(), model: model) }.value
            let quick = events.events.filter { $0["event"] as? String == "validation" && $0["scope"] as? String == "quick" }
            if configuration.every == 0 || configuration.quick == 0 { XCTAssertTrue(quick.isEmpty) }
            else {
                XCTAssertEqual(quick.count, 1)
                XCTAssertEqual(quick.first?["sample_count"] as? Int, 4)
            }
            let started = try XCTUnwrap(events.events.first { $0["event"] as? String == "update_started" })
            XCTAssertEqual(started["current_update"] as? Int, 1)
            XCTAssertEqual(started["completed_updates"] as? Int, 0)
            XCTAssertEqual(started["requested_updates"] as? Int, 1)
            XCTAssertEqual(started["epoch"] as? Int, 1)
            XCTAssertEqual(started["total_epochs"] as? Int, 1)
            XCTAssertEqual(started["sample_position"] as? Int, 1)
            let numeric = events.events.filter { $0["event"] as? String == "operation_progress" && $0["phase"] as? String == "training" }
            XCTAssertTrue(numeric.contains { $0["operation"] as? String == "Forward pass" })
            XCTAssertTrue(numeric.contains { $0["operation"] as? String == "Computing loss gradients" },
                "The fixture's final layer receives backward gradients through the loss stage.")
            XCTAssertTrue(numeric.contains { $0["operation"] as? String == "Applying optimizer update" })
            let export = try XCTUnwrap(events.events.first { $0["event"] as? String == "export_started" })
            XCTAssertEqual(export["completed_updates"] as? Int, 1)
            XCTAssertEqual(export["workflow_phase"] as? Int, 4)
            XCTAssertEqual(events.events.last?["event"] as? String, "training_completed")
        }
    }

    func testDisabledValidationDoesNotRunValidationOrEmitAnEmptyCropWarning() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset"), output = root.appendingPathComponent("output")
        try datasetFixture(dataset, validationEnabled: false)
        let model = try trainerModel(), events = Recorder()
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
            "--size", "256", "--updates-per-map", "1", "--validation-every", "1"])
        let text = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: events.append, control: .init(), model: model) }.value
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        XCTAssertEqual((result["baseline_validation"] as? [String: Any])?["status"] as? String, "disabled")
        XCTAssertFalse(events.events.contains { ($0["event"] as? String ?? "").hasPrefix("validation") })
        XCTAssertEqual(events.updates, 1)
    }

    func testControlKeepsSnapshotSaveAndAbortSemanticsIndependent() throws {
        let control = NativeMaterialTrainingControl()
        XCTAssertFalse(control.consumeCheckpoint()); XCTAssertFalse(control.shouldStopAndSave)
        control.saveCheckpoint(); XCTAssertTrue(control.consumeCheckpoint()); XCTAssertFalse(control.consumeCheckpoint())
        control.stopAndSave(); XCTAssertTrue(control.shouldStopAndSave); XCTAssertNoThrow(try control.check())
        control.stop(); XCTAssertThrowsError(try control.check()) { XCTAssertTrue($0 is CancellationError) }
    }
    func testEXRStoresEveryFloat32BitAndTopLeftChannelWithoutNormalization() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let words: [UInt32] = [0x80000000, 1, 0x3f800001, 0xbf000000, 0x40000001, 0x00800001,
            0x3e000003, 0xbe000004, 0x3f400005, 0x3e800006, 0x3fa00007, 0xbf800008,
            0x3f000009, 0xbe80000a, 0x3fc0000b, 0x3d00000c, 0x4020000d, 0xc010000e]
        let values = words.map(Float.init(bitPattern:)), url = root.appendingPathComponent("numeric.exr")
        try NativeMaterialNumericExporter.writeEXR(.init(width: 3, height: 2, channels: 3, values: values), to: url)
        let bytes = try Data(contentsOf: url)
        XCTAssertEqual(u32(bytes, 0), 20_000_630); XCTAssertEqual(u32(bytes, 4), 2)
        var cursor = 8, attributes: [String: (String, Data)] = [:]
        while bytes[cursor] != 0 {
            let name = cString(bytes, &cursor), type = cString(bytes, &cursor), count = Int(u32(bytes, cursor)); cursor += 4
            attributes[name] = (type, bytes.subdata(in: cursor..<cursor + count)); cursor += count
        }
        cursor += 1
        XCTAssertEqual(attributes["compression"]?.1, Data([0])); XCTAssertEqual(attributes["channels"]?.0, "chlist")
        let channelData = try XCTUnwrap(attributes["channels"]?.1)
        var channelCursor = 0, names: [String] = []
        while channelData[channelCursor] != 0 {
            names.append(cString(channelData, &channelCursor)); XCTAssertEqual(u32(channelData, channelCursor), 2); channelCursor += 16
        }
        XCTAssertEqual(names, ["B", "G", "R"])
        for row in 0..<2 {
            let offset = Int(u64(bytes, cursor + row * 8))
            XCTAssertEqual(u32(bytes, offset), UInt32(row)); XCTAssertEqual(u32(bytes, offset + 4), 36)
            for (stored, original) in [2, 1, 0].enumerated() {
                for column in 0..<3 { XCTAssertEqual(u32(bytes, offset + 8 + (stored * 3 + column) * 4), words[original * 6 + row * 3 + column]) }
            }
        }
    }
    func testRealTrainerUpdatesValidatesSnapshotsAndStopsWithCompleteNativePackage() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset"), output = root.appendingPathComponent("trained")
        try datasetFixture(dataset)
        let fixture = NativeMaterialModelFixture(), original = try fixture.model(adapter: true)
        var configuration: [String: Any] = ["schema": "texture-studio-material-lora-v1", "architecture": "pbrnxt-native-v1",
            "target": "height", "scope": "final-map", "step": 0, "training_size": 256,
            "image_padding": false, "image_resizing": false, "base": ["sha256": String(repeating: "a", count: 64)]]
        configuration["layers"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(original.layers))
        let model = try NativeMaterialModel(baseWeights: original.baseWeights, adapterWeights: original.adapterWeights,
            layers: original.layers, configuration: configuration, baseSHA256: String(repeating: "a", count: 64), architecture: .test)
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
            "--size", "256", "--updates-per-map", "5", "--validation-every", "2", "--learning-rate", "0.001",
            "--model-name", "  石 Stone / Displacement  "])
        let control = NativeMaterialTrainingControl(), events = Recorder()
        let text = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: { line in
            events.append(line)
            if line.contains("\"event\":\"update\"") {
                if events.updates == 1 { control.saveCheckpoint() } else { control.stopAndSave() }
            }
        }, control: control, model: model) }.value
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        XCTAssertEqual(result["status"] as? String, "stopped"); XCTAssertEqual(result["completed_updates"] as? Int, 2)
        XCTAssertEqual(result["stopped_reason"] as? String, "user_stop")
        XCTAssertEqual(result["requested_updates"] as? Int, 5)
        XCTAssertEqual(result["model_name"] as? String, "石 Stone / Displacement")
        let recordedRun = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("run.json"))) as? [String: Any])
        XCTAssertEqual(recordedRun["model_name"] as? String, "石 Stone / Displacement")
        let saved = try NativeSafetensors(contentsOf: output.appendingPathComponent("export/adapter.safetensors"))
        XCTAssertNotEqual(try saved.tensorBytes(named: "ups.3.model.10.lora_B"), original.adapterWeights["ups.3.model.10.lora_B"]!.bytes)
        XCTAssertNoThrow(try NativeMaterialPackage.verify(output.appendingPathComponent("export")))
        XCTAssertTrue(FileManager.default.fileExists(atPath: output.appendingPathComponent("checkpoint-step-00000001.safetensors").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: output.appendingPathComponent("checkpoint-step-00000002.safetensors").path))
        for path in ["checkpoint-step-00000001.safetensors", "checkpoint-step-00000002.safetensors", "export/adapter.safetensors"] {
            let checkpoint = try WorkbenchResult.decode(WorkbenchCheckpoint.self,
                output: NativeMaterialCheckpoint.inspect(at: output.appendingPathComponent(path)))
            XCTAssertEqual(checkpoint.modelName, "石 Stone / Displacement")
            XCTAssertTrue(checkpoint.title.hasPrefix("石 Stone / Displacement · "))
        }
        let packageConfig = try NativeMaterialTransfer.object(output.appendingPathComponent("export/config.json"))
        XCTAssertEqual(packageConfig["model_name"] as? String, "石 Stone / Displacement")
        XCTAssertTrue(try String(contentsOf: output.appendingPathComponent("export/README.md"), encoding: .utf8)
            .contains("Model name: 石 Stone / Displacement"))
        XCTAssertEqual(events.updates, 2)
        XCTAssertFalse(events.executedOnMain)
        let final = try XCTUnwrap(result["final_validation"] as? [String: Any])
        XCTAssertEqual(final["scope"] as? String, "full"); XCTAssertEqual(final["sample_count"] as? Int, 1)
    }
    func testAbortBeforeTrainingDoesNotCreateOutput() throws {
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", root.path, "--output", root.appendingPathComponent("cancelled").path])
        let control = NativeMaterialTrainingControl(); control.stop()
        XCTAssertThrowsError(try NativeMaterialTrainer.train(options, onEvent: { _ in }, control: control)) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertFalse(FileManager.default.fileExists(atPath: options.output.path))
    }
    func testAbortFromFinalValidationPreventsCheckpointAndExport() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset"), output = root.appendingPathComponent("aborted-before-save")
        try datasetFixture(dataset)
        let model = try trainerModel()
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
            "--size", "256", "--updates-per-map", "1", "--validation-every", "2", "--learning-rate", "0.001"])
        let control = NativeMaterialTrainingControl(), events = Recorder()
        do {
            _ = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: { line in
                events.append(line)
                if line.contains("\"event\":\"validation\""), events.updates == 1 { control.stop() }
            }, control: control, model: model) }.value
            XCTFail("Abort from final validation must prevent checkpoint and package publication")
        } catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(events.updates, 1)
        XCTAssertFalse(events.events.contains { $0["event"] as? String == "checkpoint_saved" })
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("checkpoint-step-00000001.safetensors").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("export").path))
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("run.json"))) as? [String: Any])
        XCTAssertEqual(result["status"] as? String, "aborted")
        XCTAssertEqual(result["completed_updates"] as? Int, 1)
    }
    func testAbortFromFinalCheckpointEventKeepsCheckpointAndPreventsExport() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset"), output = root.appendingPathComponent("aborted")
        try datasetFixture(dataset)
        let original = try NativeMaterialModelFixture().model(adapter: true)
        var configuration: [String: Any] = ["schema": "texture-studio-material-lora-v1", "architecture": "pbrnxt-native-v1",
            "target": "height", "scope": "final-map", "step": 0, "training_size": 256,
            "image_padding": false, "image_resizing": false, "base": ["sha256": String(repeating: "a", count: 64)]]
        configuration["layers"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(original.layers))
        let model = try NativeMaterialModel(baseWeights: original.baseWeights, adapterWeights: original.adapterWeights,
            layers: original.layers, configuration: configuration, baseSHA256: String(repeating: "a", count: 64), architecture: .test)
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
            "--size", "256", "--updates-per-map", "1", "--validation-every", "2", "--learning-rate", "0.001"])
        let control = NativeMaterialTrainingControl(), events = Recorder()
        do {
            _ = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: { line in
                events.append(line)
                if line.contains("\"event\":\"checkpoint_saved\"") { control.stop() }
            }, control: control, model: model) }.value
            XCTFail("Abort after the final checkpoint must prevent package publication")
        } catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertEqual(events.updates, 1)
        let checkpoint = output.appendingPathComponent("checkpoint-step-00000001.safetensors")
        XCTAssertNoThrow(try NativeMaterialCheckpoint.inspect(at: checkpoint))
        let saved = try NativeSafetensors(contentsOf: checkpoint)
        XCTAssertNotEqual(try saved.tensorBytes(named: "ups.3.model.10.lora_B"), original.adapterWeights["ups.3.model.10.lora_B"]!.bytes)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("export").path))
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("run.json"))) as? [String: Any])
        XCTAssertEqual(result["status"] as? String, "aborted")
        XCTAssertEqual(result["completed_updates"] as? Int, 1)
    }
    func testTimeLimitAfterBaselineStartsNoUpdateAndSavesUnchangedCheckpoint() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset"), output = root.appendingPathComponent("timed-out-before-update")
        try datasetFixture(dataset)
        let model = try trainerModel(), original = model.adapterWeights
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
            "--size", "256", "--updates-per-map", "3", "--max-minutes", "1"])
        let clock = ManualTrainingClock(), events = Recorder()
        let text = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: { line in
            events.append(line)
            if line.contains("\"event\":\"validation\"") { clock.advanceOnce(by: .seconds(61)) }
        }, control: .init(), model: model, now: { clock.now }) }.value
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        XCTAssertEqual(result["status"] as? String, "stopped")
        XCTAssertEqual(result["stopped_reason"] as? String, "time_limit")
        XCTAssertEqual(result["completed_updates"] as? Int, 0)
        XCTAssertEqual(result["requested_updates"] as? Int, 3)
        XCTAssertEqual(result["training_performed"] as? Bool, false)
        XCTAssertEqual(result["elapsed_training_seconds"] as? Double, 61)
        XCTAssertEqual(events.updates, 0, "An expired baseline must not start another whole-grid training step")
        let stopped = try XCTUnwrap(events.events.first { $0["event"] as? String == "training_stopped" })
        XCTAssertEqual(stopped["stopped_reason"] as? String, "time_limit")
        let saved = try NativeSafetensors(contentsOf: output.appendingPathComponent("checkpoint-step-00000000.safetensors"))
        for (name, weight) in original { XCTAssertEqual(try saved.tensorBytes(named: name), weight.bytes, name) }
        XCTAssertNoThrow(try NativeMaterialPackage.verify(output.appendingPathComponent("export")))
        let recorded = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("run.json"))) as? [String: Any])
        XCTAssertEqual(recorded["status"] as? String, "stopped")
        XCTAssertEqual(recorded["stopped_reason"] as? String, "time_limit")
    }
    func testTimeLimitDuringUpdateFinishesCurrentStepAndReportsDuration() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset"), output = root.appendingPathComponent("timed-out-after-update")
        try datasetFixture(dataset, trainingInputCode: 126)
        let model = try trainerModel(), original = model.adapterWeights
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
            "--size", "256", "--updates-per-map", "3", "--max-minutes", "1", "--learning-rate", "0.001"])
        let clock = ManualTrainingClock(), events = Recorder()
        let text = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: { line in
            events.append(line)
            if line.contains("\"phase\":\"training\"") { clock.advanceOnce(by: .seconds(61)) }
        }, control: .init(), model: model, now: { clock.now }) }.value
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        XCTAssertEqual(result["status"] as? String, "stopped")
        XCTAssertEqual(result["stopped_reason"] as? String, "time_limit")
        XCTAssertEqual(result["completed_updates"] as? Int, 1)
        XCTAssertEqual(result["requested_updates"] as? Int, 3)
        XCTAssertEqual(events.updates, 1, "The in-flight update completes, but no next step starts after the deadline")
        let update = try XCTUnwrap(events.events.first { $0["event"] as? String == "update" })
        XCTAssertEqual(update["update_duration_seconds"] as? Double, 61)
        XCTAssertEqual(update["elapsed_training_seconds"] as? Double, 61)
        let saved = try NativeSafetensors(contentsOf: output.appendingPathComponent("checkpoint-step-00000001.safetensors"))
        XCTAssertNotEqual(try saved.tensorBytes(named: "ups.3.model.10.lora_B"), original["ups.3.model.10.lora_B"]!.bytes)
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("checkpoint-step-00000002.safetensors").path))
        XCTAssertNoThrow(try NativeMaterialPackage.verify(output.appendingPathComponent("export")))
    }
    func testAllRequestedUpdatesRemainCompletedWhenLastStepCrossesDeadline() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset"), output = root.appendingPathComponent("completed-at-deadline")
        try datasetFixture(dataset, trainingInputCode: 126)
        let model = try trainerModel(), clock = ManualTrainingClock(), events = Recorder()
        let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
            "--size", "256", "--updates-per-map", "1", "--max-minutes", "1"])
        let text = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: { line in
            events.append(line)
            if line.contains("\"phase\":\"training\"") { clock.advanceOnce(by: .seconds(61)) }
        }, control: .init(), model: model, now: { clock.now }) }.value
        let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
        XCTAssertEqual(result["status"] as? String, "completed")
        XCTAssertNil(result["stopped_reason"])
        XCTAssertEqual(result["completed_updates"] as? Int, 1)
        XCTAssertEqual(result["requested_updates"] as? Int, 1)
        XCTAssertEqual(events.updates, 1)
        XCTAssertEqual(result["elapsed_training_seconds"] as? Double, 61)
        let update = try XCTUnwrap(events.events.first { $0["event"] as? String == "update" })
        XCTAssertEqual(update["update_duration_seconds"] as? Double, 61)
        XCTAssertTrue(events.events.contains { $0["event"] as? String == "feature_progress" && $0["phase"] as? String == "training" })
        XCTAssertFalse(events.events.contains { $0["event"] as? String == "training_stopped" })
    }
    func testStopDuringNumericStepFinishesItAndAbortDiscardsUnfinishedStep() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil)
        let root = try temporary(); defer { try? FileManager.default.removeItem(at: root) }
        let dataset = root.appendingPathComponent("dataset")
        try datasetFixture(dataset, trainingInputCode: 126)
        for abort in [false, true] {
            let output = root.appendingPathComponent(abort ? "abort-in-step" : "stop-in-step")
            let model = try trainerModel(), original = model.adapterWeights
            let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
                "--size", "256", "--updates-per-map", "3", "--learning-rate", "0.001"])
            let control = NativeMaterialTrainingControl(), events = Recorder()
            do {
                let text = try await Task.detached { try NativeMaterialTrainer.train(options, onEvent: { line in
                    events.append(line)
                    if line.contains("\"phase\":\"training\"") {
                        if abort { control.stop() } else { control.stopAndSave() }
                    }
                }, control: control, model: model) }.value
                XCTAssertFalse(abort, "Immediate Abort must cancel before the adapter update is committed")
                let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
                XCTAssertEqual(result["status"] as? String, "stopped")
                XCTAssertEqual(result["stopped_reason"] as? String, "user_stop")
                XCTAssertEqual(result["completed_updates"] as? Int, 1)
                XCTAssertEqual(events.updates, 1)
                let saved = try NativeSafetensors(contentsOf: output.appendingPathComponent("checkpoint-step-00000001.safetensors"))
                XCTAssertNotEqual(try saved.tensorBytes(named: "ups.3.model.10.lora_B"), original["ups.3.model.10.lora_B"]!.bytes)
                XCTAssertNoThrow(try NativeMaterialPackage.verify(output.appendingPathComponent("export")))
            } catch {
                XCTAssertTrue(abort, "Graceful Stop must finish its active numeric step")
                XCTAssertTrue(error is CancellationError)
                XCTAssertEqual(events.updates, 0)
                for (name, weight) in original { XCTAssertEqual(model.adapterWeights[name]?.bytes, weight.bytes, name) }
                XCTAssertFalse(FileManager.default.fileExists(atPath: output.appendingPathComponent("export").path))
                let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("run.json"))) as? [String: Any])
                XCTAssertEqual(result["status"] as? String, "aborted")
                XCTAssertEqual(result["completed_updates"] as? Int, 0)
            }
        }
    }
    /// Opt in on the installed learned base and real maps. The ordinary small
    /// graph fixtures cannot detect production-grid compilation/activation
    /// runaway. Real pairs measure numeric execution, not material quality.
    func testProductionExactGridTrainingResourceRegression() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["TEXTURE_STUDIO_PRODUCTION_TRAINING_BENCHMARK"] == "1" else {
            throw XCTSkip("Set TEXTURE_STUDIO_PRODUCTION_TRAINING_BENCHMARK=1 to train the pinned learned base on real 1K/2K maps.")
        }
        guard let device = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal is unavailable.") }
        let base = URL(fileURLWithPath: environment["TEXTURE_STUDIO_BENCHMARK_BASE_CHECKPOINT"] ??
            FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
                .appendingPathComponent("Texture Studio/Material Models/pbrnxt-base/" + NativeMaterialModel.pinnedFilename).path)
        let source = URL(fileURLWithPath: environment["TEXTURE_STUDIO_BENCHMARK_SOURCE_DIRECTORY"] ?? "/opt/ipde/sources_mats/white_stucco_02")
        let pairCount = Int(environment["TEXTURE_STUDIO_PRODUCTION_TRAINING_PAIRS"] ?? "1") ?? 1
        guard (1...4).contains(pairCount) else { throw StudioError("Production benchmark supports 1 through 4 distinct real material pairs.") }
        var availablePairs = [ProductionPair(id: "white_stucco_02", directory: source)]
        for name in ["red_brick", "pine_bark", "concrete_layers"] {
            let directory = name == "red_brick" ? environment["TEXTURE_STUDIO_BENCHMARK_SECOND_SOURCE_DIRECTORY"] : nil
            availablePairs.append(ProductionPair(id: name, directory: URL(fileURLWithPath: directory ?? "/opt/ipde/sources_mats/" + name)))
        }
        let pairs = Array(availablePairs.prefix(pairCount))
        let sourceFiles = pairs.flatMap { [$0.input, $0.target] }
        for file in [base] + sourceFiles { XCTAssertTrue(FileManager.default.fileExists(atPath: file.path), "Missing benchmark input: \(file.path)") }
        let sizes = (environment["TEXTURE_STUDIO_PRODUCTION_TRAINING_SIZES"] ?? "1024,2048").split(separator: ",").compactMap { Int($0) }
        XCTAssertFalse(sizes.isEmpty)
        XCTAssertTrue(sizes.allSatisfy { [256, 512, 1024, 2048].contains($0) })
        let updates = Int(environment["TEXTURE_STUDIO_PRODUCTION_TRAINING_UPDATES"] ?? "1") ?? 1
        XCTAssertGreaterThan(updates, 0)
        let expectedUpdates = updates * pairCount
        let maxMinutes = 90
        let scope = environment["TEXTURE_STUDIO_PRODUCTION_TRAINING_SCOPE"] ?? "final-map"
        XCTAssertTrue(["final-map", "map-decoder"].contains(scope))
        let rank = Int(environment["TEXTURE_STUDIO_PRODUCTION_TRAINING_RANK"] ?? "8") ?? 8
        let alpha = Float(environment["TEXTURE_STUDIO_PRODUCTION_TRAINING_ALPHA"] ?? "8") ?? 8
        XCTAssertTrue((1...64).contains(rank))
        XCTAssertTrue(alpha.isFinite && alpha > 0)
        let repository = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let reports = URL(fileURLWithPath: environment["TEXTURE_STUDIO_BENCHMARK_OUTPUT_DIRECTORY"] ??
            repository.appendingPathComponent("out/native-training-regression").path)
            .appendingPathComponent("run-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: reports, withIntermediateDirectories: true)
        let sourceHashes = try sourceFiles.map { try NativeMaterialTrainer.checksum(Data(contentsOf: $0, options: .mappedIfSafe)) }
        for size in sizes {
            let temporary = try temporary(); defer { try? FileManager.default.removeItem(at: temporary) }
            let dataset = temporary.appendingPathComponent("dataset")
            let inputHashes = try productionDatasetFixture(dataset, pairs: pairs, size: size)
            XCTAssertEqual(Set(inputHashes).count, pairCount, "Distinct material pixels must exercise frozen-prefix cache eviction.")
            let output = reports.appendingPathComponent("trained-\(size)")
            let options = try NativeMaterialTrainer.Options(["train", "--dataset", dataset.path, "--output", output.path,
                "--base-checkpoint", base.path, "--size", String(size), "--scope", scope, "--updates-per-map", String(updates),
                "--validation-every", "20", "--lora-rank", String(rank), "--lora-alpha", String(alpha), "--learning-rate", "0.00001", "--max-minutes", String(maxMinutes)])
            let control = NativeMaterialTrainingControl()
            let metrics = ProductionMetrics(device: device, control: control, volumeURL: reports)
            let started = ContinuousClock.now
            metrics.start()
            defer { metrics.stop() }
            let completed: (String, [String: Data])
            do {
                completed = try await Task.detached {
                    try autoreleasepool {
                        let model = try NativeMaterialModel.load(checkpointURL: nil, baseURL: base,
                            target: "height", scope: scope, rank: rank, alpha: alpha, training: true)
                        let originalFactors = model.adapterWeights.mapValues(\.bytes)
                        let text = try NativeMaterialTrainer.train(options, onEvent: { metrics.record($0) },
                            control: control, model: model)
                        return (text, originalFactors)
                    }
                }.value
            } catch {
                metrics.stop()
                let failure: [String: Any] = ["size": size, "status": "failed", "error": error.localizedDescription,
                    "scope": scope, "rank": rank, "alpha": Double(alpha), "training_pair_count": pairCount,
                    "updates_per_map": updates, "max_minutes": maxMinutes,
                    "training_input_sha256": inputHashes, "source_sha256": sourceHashes,
                    "resources": metrics.snapshot, "events": metrics.events]
                try JSONSerialization.data(withJSONObject: failure, options: [.sortedKeys, .prettyPrinted])
                    .write(to: reports.appendingPathComponent("training-\(size)-failed.json"), options: .atomic)
                print("PRODUCTION_TRAINING_FAILED report=\(reports.path); resources=\(metrics.snapshot)")
                throw error
            }
            let (text, originalFactors) = completed
            metrics.stop()
            let finalResources = metrics.snapshot
            let resourceLimit = try XCTUnwrap(finalResources["benchmark_resource_abort_limit_bytes"] as? UInt64)
            XCTAssertEqual(finalResources["benchmark_resource_abort_requested"] as? Bool, false,
                           "Crossing the resource guard must fail even if it occurs after the last update.")
            XCTAssertLessThanOrEqual(try XCTUnwrap(finalResources["process_peak_physical_footprint_bytes"] as? UInt64), resourceLimit)
            XCTAssertLessThanOrEqual(try XCTUnwrap(finalResources["sampled_peak_metal_allocation_bytes"] as? UInt64), resourceLimit)
            let duration = started.duration(to: .now).components
            let elapsed = Double(duration.seconds) + Double(duration.attoseconds) / 1e18
            let result = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(text.utf8)) as? [String: Any])
            let actualUpdates = result["completed_updates"] as? Int ?? 0
            XCTAssertEqual(result["status"] as? String, "completed")
            XCTAssertEqual(actualUpdates, expectedUpdates)
            XCTAssertEqual(result["native_dimensions"] as? [Int], [size, size])
            XCTAssertEqual(result["image_padding"] as? Bool, false)
            XCTAssertEqual(result["image_resizing"] as? Bool, false)
            let package = output.appendingPathComponent("export")
            XCTAssertNoThrow(try NativeMaterialPackage.verify(package))
            let saved = try NativeSafetensors(contentsOf: package.appendingPathComponent("adapter.safetensors"))
            let configuration = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(saved.metadata["configuration"]!.utf8)) as? [String: Any])
            XCTAssertEqual(configuration["training_size"] as? Int, size)
            XCTAssertEqual(configuration["step"] as? Int, expectedUpdates)
            XCTAssertEqual(configuration["scope"] as? String, scope)
            XCTAssertEqual((configuration["base"] as? [String: Any])?["sha256"] as? String, NativeMaterialModel.pinnedSHA256,
                "The production benchmark must run the actual pinned learned base.")
            let layerRecords = try XCTUnwrap(configuration["layers"] as? [String: [String: Any]])
            for (name, layer) in layerRecords {
                XCTAssertEqual(layer["rank"] as? Int, rank, name)
                XCTAssertEqual(layer["alpha"] as? Double, Double(alpha), name)
            }
            XCTAssertGreaterThan(originalFactors.count, 100, "Production training must preserve the complete selected adapter scope.")
            XCTAssertEqual(Set(saved.tensors.keys), Set(originalFactors.keys))
            XCTAssertTrue(try originalFactors.keys.contains { name in
                try name.hasSuffix(".lora_B") && saved.tensorBytes(named: name) != originalFactors[name]!
            }, "The actual production LoRA factors must change.")
            for name in ["ups.3.model.0.lora_B", "ups.3.model.1.sub.0.RDB1.conv1.0.lora_B", "ups.3.model.10.lora_B"] {
                let original = try XCTUnwrap(originalFactors[name], "Missing early/late production adapter: \(name)")
                XCTAssertNotEqual(try saved.tensorBytes(named: name), original, "Gradients must reach the complete selected branch: \(name)")
            }
            if scope == "map-decoder" {
                let name = "gen.m_dec_3.m_up3.0.up.1.lora_B"
                XCTAssertNotEqual(try saved.tensorBytes(named: name), try XCTUnwrap(originalFactors[name]), "Gradients must reach decoder upsampling.")
            }
            let baseline = try XCTUnwrap(result["baseline_validation"] as? [String: Any])
            let final = try XCTUnwrap(result["final_validation"] as? [String: Any])
            XCTAssertEqual(baseline["sample_count"] as? Int, 1)
            XCTAssertEqual(final["sample_count"] as? Int, 1)
            XCTAssertTrue((baseline["mae"] as? Double)?.isFinite == true)
            XCTAssertTrue((final["mae"] as? Double)?.isFinite == true)
            let updateEvents = metrics.events.filter { $0["event"] as? String == "update" }
            XCTAssertEqual(updateEvents.count, expectedUpdates)
            let trainingIDs = pairs.indices.map { pairCount == 1 ? "train" : "train-" + pairs[$0].id }
            for id in trainingIDs {
                XCTAssertEqual(updateEvents.filter { $0["sample_id"] as? String == id }.count, updates, "Each real material must recur in the shuffled training passes.")
            }
            for (index, file) in sourceFiles.enumerated() {
                XCTAssertEqual(try NativeMaterialTrainer.checksum(Data(contentsOf: file, options: .mappedIfSafe)), sourceHashes[index])
            }
            let report: [String: Any] = ["size": size, "elapsed_seconds": elapsed,
                "source_directory": source.path, "base_checkpoint": base.path,
                "base_sha256": NativeMaterialModel.pinnedSHA256, "source_sha256": sourceHashes,
                "scope": scope, "rank": rank, "alpha": Double(alpha), "completed_updates": actualUpdates,
                "requested_updates": expectedUpdates, "status": result["status"] as? String ?? "unknown",
                "training_pair_count": pairCount, "updates_per_map": updates, "max_minutes": maxMinutes, "training_input_sha256": inputHashes,
                "source_pairs": pairs.map { ["material_id": $0.id, "input": $0.input.path, "target": $0.target.path] },
                "measurement_scope": "Real learned base, exact native pixels, distinct real training materials in repeated shuffled passes and the first pair repeated for validation; numeric execution and resource regression, not material quality.",
                "resource_measurement_scope": "RSS and physical-footprint peaks are process lifetime high-water marks; Metal peak is sampled for this run. Later sizes may include an earlier process peak.",
                "resources": finalResources, "events": metrics.events, "run": result]
            let bytes = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys, .prettyPrinted])
            try bytes.write(to: reports.appendingPathComponent("training-\(size).json"), options: .atomic)
            print("PRODUCTION_TRAINING_METRICS \(size): \(String(decoding: try JSONSerialization.data(withJSONObject: metrics.snapshot, options: [.sortedKeys]), as: UTF8.self)); elapsed_seconds=\(elapsed); report=\(reports.path)")
        }
    }
    private final class ProductionMetrics: @unchecked Sendable {
        private let lock = NSLock(), device: MTLDevice, control: NativeMaterialTrainingControl
        private let footprintLimit = ProcessInfo.processInfo.physicalMemory * 9 / 10
        private var resourceAbortRequested = false
        private let started = ProcessInfo.processInfo.systemUptime
        private var timer: DispatchSourceTimer?
        private var maximumMetal: UInt64 = 0, maximumResident: UInt64 = 0, peakFootprint: UInt64 = 0
        private var currentResident: UInt64 = 0, currentFootprint: UInt64 = 0, currentMetal: UInt64 = 0
        private var recorded: [[String: Any]] = []
        private let volumeURL: URL
        private let initialSwapBytes: UInt64?, initialAvailableDiskBytes: UInt64?
        init(device: MTLDevice, control: NativeMaterialTrainingControl, volumeURL: URL) {
            self.device = device; self.control = control; self.volumeURL = volumeURL
            initialSwapBytes = Self.systemSwapUsedBytes()
            initialAvailableDiskBytes = Self.availableDiskBytes(at: volumeURL)
        }
        func start() {
            let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "production-training-resource-sampler"))
            timer.schedule(deadline: .now(), repeating: .milliseconds(100))
            timer.setEventHandler { [weak self] in self?.sample() }
            self.timer = timer; timer.resume(); sample()
        }
        func stop() { timer?.cancel(); timer = nil; sample() }
        func sample() {
            var usage = rusage(); _ = getrusage(RUSAGE_SELF, &usage)
            var info = task_vm_info_data_t()
            var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
            let status = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                    task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
                }
            }
            let metal = UInt64(device.currentAllocatedSize)
            let abort = lock.withLock {
                maximumMetal = max(maximumMetal, metal)
                maximumResident = max(maximumResident, UInt64(max(0, usage.ru_maxrss)))
                currentMetal = metal
                if status == KERN_SUCCESS {
                    peakFootprint = max(peakFootprint, UInt64(max(0, info.ledger_phys_footprint_peak)))
                    currentResident = info.resident_size; currentFootprint = info.phys_footprint
                }
                if max(currentFootprint, currentMetal) > footprintLimit {
                    resourceAbortRequested = true
                    return true
                }
                return false
            }
            if abort { control.stop() }
        }
        func record(_ text: String) {
            sample()
            guard let data = text.data(using: .utf8), var event = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
            event["elapsed_seconds"] = ProcessInfo.processInfo.systemUptime - started
            event["resources"] = snapshot
            lock.withLock { recorded.append(event) }
            let annotated = (try? JSONSerialization.data(withJSONObject: event, options: [.sortedKeys])) ?? data
            print("PRODUCTION_TRAINING_EVENT " + String(decoding: annotated, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines))
        }
        var snapshot: [String: Any] {
            // Storage queries run only when an event/report requests a snapshot,
            // rather than adding filesystem work to the 100 ms memory sampler.
            let swapBytes = Self.systemSwapUsedBytes(), availableDiskBytes = Self.availableDiskBytes(at: volumeURL)
            var result: [String: Any] = lock.withLock { ["process_maximum_rss_bytes": maximumResident,
                "process_peak_physical_footprint_bytes": peakFootprint, "sampled_peak_metal_allocation_bytes": maximumMetal,
                "current_resident_bytes": currentResident, "current_physical_footprint_bytes": currentFootprint,
                "current_metal_allocation_bytes": currentMetal,
                "benchmark_resource_abort_limit_bytes": footprintLimit, "benchmark_resource_abort_requested": resourceAbortRequested,
                "metal_sample_interval_ms": 100, "physical_memory_bytes": ProcessInfo.processInfo.physicalMemory,
                "metal_recommended_working_set_bytes": device.recommendedMaxWorkingSetSize] }
            result["initial_system_swap_used_bytes"] = initialSwapBytes.map { $0 as Any } ?? NSNull()
            result["system_swap_used_bytes"] = swapBytes.map { $0 as Any } ?? NSNull()
            result["system_swap_used_delta_bytes"] = Self.signedDelta(swapBytes, from: initialSwapBytes).map { $0 as Any } ?? NSNull()
            result["initial_volume_available_capacity_bytes"] = initialAvailableDiskBytes.map { $0 as Any } ?? NSNull()
            result["volume_available_capacity_bytes"] = availableDiskBytes.map { $0 as Any } ?? NSNull()
            result["volume_available_capacity_delta_bytes"] = Self.signedDelta(availableDiskBytes, from: initialAvailableDiskBytes).map { $0 as Any } ?? NSNull()
            result["volume_measurement_path"] = volumeURL.path
            result["storage_measurement_scope"] = "Swap is system-wide and available capacity is volume-wide. Deltas can include activity outside this training process."
            return result
        }
        private static func systemSwapUsedBytes() -> UInt64? {
            var usage = xsw_usage(), size = MemoryLayout<xsw_usage>.size
            guard sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 else { return nil }
            return usage.xsu_used
        }
        private static func availableDiskBytes(at url: URL) -> UInt64? {
            // URL resource values may cache capacity across snapshots. Ask the
            // filesystem each time so growth on this volume remains observable.
            guard let attributes = try? FileManager.default.attributesOfFileSystem(forPath: url.path),
                  let available = attributes[.systemFreeSize] as? NSNumber,
                  available.int64Value >= 0 else { return nil }
            return available.uint64Value
        }
        private static func signedDelta(_ current: UInt64?, from initial: UInt64?) -> Int64? {
            guard let current, let initial, let currentSigned = Int64(exactly: current),
                  let initialSigned = Int64(exactly: initial) else { return nil }
            return currentSigned - initialSigned
        }
        var events: [[String: Any]] { lock.withLock { recorded } }
    }
    private struct ProductionPair: Sendable {
        let id: String, input: URL, target: URL
        init(id: String, directory: URL) {
            self.id = id
            input = directory.appendingPathComponent(id + "_diff_2k.png")
            target = directory.appendingPathComponent(id + "_disp_2k.png")
        }
    }
    private func productionDatasetFixture(_ root: URL, pairs: [ProductionPair], size: Int) throws -> [String] {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        var entries: [[String: Any]] = [], inputHashes: [String] = []
        for (pairIndex, pair) in pairs.enumerated() {
            var files: [String: URL] = [:], hashes: [String: String] = [:]
            for (role, original) in [("input", pair.input), ("height", pair.target)] {
                let header = try NativePNG.inspect(original)
                XCTAssertEqual(header.bits, 16)
                XCTAssertGreaterThanOrEqual(header.width, size); XCTAssertGreaterThanOrEqual(header.height, size)
                let file: URL
                if header.width == size, header.height == size { file = original }
                else {
                    file = root.appendingPathComponent(pair.id + "-" + role + ".png")
                    let rectangle = [(header.width - size) / 2, (header.height - size) / 2, size, size]
                    try NativePNG.crop(original, rectangle: rectangle).encoded().write(to: file)
                }
                files[role] = file
                hashes[role] = try NativeMaterialTrainer.checksum(Data(contentsOf: file, options: .mappedIfSafe))
            }
            inputHashes.append(hashes["input"]!)
            let splits = pairIndex == 0 ? ["train", "validation"] : ["train"]
            for split in splits {
                let id = split == "validation" || pairs.count == 1 ? split : "train-" + pair.id
                let folder = root.appendingPathComponent("samples/" + id)
                try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
                var metadata: [String: [String: Any]] = [:]
                for role in ["input", "height"] {
                    metadata[role] = ["filename": files[role]!.path, "storage": "source_reference", "encoding": role == "input" ? "srgb" : "linear_data",
                        "sample_sha256": hashes[role]!, "source": ["path": files[role]!.path, "file_sha256": hashes[role]!]]
                }
                let sample: [String: Any] = ["sample_id": id, "material_id": id, "status": "approved", "split": split,
                    "sample_pixel_dimensions": [size, size], "maps": files.mapValues(\.path), "map_metadata": metadata]
                try JSONSerialization.data(withJSONObject: sample).write(to: folder.appendingPathComponent("sample.json"))
                entries.append(["sample_id": id, "material_id": id, "status": "approved", "split": split, "path": "samples/" + id])
            }
        }
        try JSONSerialization.data(withJSONObject: ["schema_version": 2, "samples": entries]).write(to: root.appendingPathComponent("dataset.json"))
        return inputHashes
    }
    private final class Recorder: @unchecked Sendable {
        let lock = NSLock(); private var lines: [String] = [], mainThread = false
        func append(_ line: String) { lock.withLock { lines.append(line); mainThread = mainThread || Thread.isMainThread } }
        var updates: Int { lock.withLock { lines.filter { $0.contains("\"event\":\"update\"") }.count } }
        var executedOnMain: Bool { lock.withLock { mainThread } }
        var events: [[String: Any]] { lock.withLock { lines.compactMap { try? JSONSerialization.jsonObject(with: Data($0.utf8)) as? [String: Any] } } }
    }
    private final class ManualTrainingClock: @unchecked Sendable {
        private let lock = NSLock()
        private var instant = ContinuousClock.now
        private var advanced = false
        var now: ContinuousClock.Instant { lock.withLock { instant } }
        func advanceOnce(by duration: Duration) {
            lock.withLock {
                guard !advanced else { return }
                instant = instant.advanced(by: duration)
                advanced = true
            }
        }
    }
    private func trainerModel() throws -> NativeMaterialModel {
        let original = try NativeMaterialModelFixture().model(adapter: true)
        var configuration: [String: Any] = ["schema": "texture-studio-material-lora-v1", "architecture": "pbrnxt-native-v1",
            "target": "height", "scope": "final-map", "step": 0, "training_size": 256,
            "image_padding": false, "image_resizing": false, "base": ["sha256": String(repeating: "a", count: 64)]]
        configuration["layers"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(original.layers))
        return try NativeMaterialModel(baseWeights: original.baseWeights, adapterWeights: original.adapterWeights,
            layers: original.layers, configuration: configuration, baseSHA256: String(repeating: "a", count: 64), architecture: .test)
    }
    private func datasetFixture(_ root: URL, trainingInputCode: UInt8 = 127, validationCount: Int = 1,
                                validationEnabled: Bool = true, quickCount: Int = 4, trainingCount: Int = 1) throws {
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
        var entries: [[String: Any]] = []
        let identities = (0..<trainingCount).map { (id: trainingCount == 1 ? "train" : "train-\($0)", split: "train") } + (0..<validationCount).map {
            (id: validationCount == 1 ? "validation" : "validation-\($0)", split: "validation")
        }
        for (id, split) in identities {
            let folder = root.appendingPathComponent("samples/" + id)
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
            let input = try NativePNG(header: .init(width: 256, height: 256, bits: 8, channels: 3, color: 2, interlace: 0),
                pixels: Data(repeating: split == "train" ? trainingInputCode : 127, count: 256 * 256 * 3), colorChunks: []).encoded()
            let target = try NativePNG(header: .init(width: 256, height: 256, bits: 16, channels: 1, color: 0, interlace: 0),
                pixels: Data(repeating: 128, count: 256 * 256 * 2), colorChunks: []).encoded()
            try input.write(to: folder.appendingPathComponent("diffuse.png")); try target.write(to: folder.appendingPathComponent("height.png"))
            let entry: [String: Any] = ["sample_id": id, "material_id": id, "status": "approved", "split": split, "path": "samples/" + id]
            entries.append(entry)
            let sample: [String: Any] = ["sample_id": id, "material_id": id, "status": "approved", "split": split,
                "sample_pixel_dimensions": [256, 256], "maps": ["input": "diffuse.png", "height": "height.png"],
                "map_metadata": ["input": ["filename": "diffuse.png", "encoding": "srgb", "sample_sha256": NativeMaterialTrainer.checksum(input)],
                    "height": ["filename": "height.png", "sample_sha256": NativeMaterialTrainer.checksum(target)]]]
            try JSONSerialization.data(withJSONObject: sample).write(to: folder.appendingPathComponent("sample.json"))
        }
        try JSONSerialization.data(withJSONObject: ["schema_version": 2, "samples": entries,
            "validation": ["enabled": validationEnabled, "quick_count": quickCount]]).write(to: root.appendingPathComponent("dataset.json"))
    }
    private func temporary() throws -> URL {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("native-training-test-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false); return root
    }
    private func cString(_ bytes: Data, _ cursor: inout Int) -> String {
        let start = cursor; while bytes[cursor] != 0 { cursor += 1 }
        defer { cursor += 1 }; return String(decoding: bytes[start..<cursor], as: UTF8.self)
    }
    private func u32(_ bytes: Data, _ offset: Int) -> UInt32 { (0..<4).reduce(UInt32(0)) { $0 | UInt32(bytes[offset + $1]) << ($1 * 8) } }
    private func u64(_ bytes: Data, _ offset: Int) -> UInt64 { (0..<8).reduce(UInt64(0)) { $0 | UInt64(bytes[offset + $1]) << ($1 * 8) } }
}

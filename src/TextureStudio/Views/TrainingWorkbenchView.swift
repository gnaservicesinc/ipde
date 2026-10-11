import SwiftUI
import AppKit

struct TrainingWorkbenchView: View {
    @Bindable var store: WorkbenchStore
    @State private var showCheckpoints = false
    @AppStorage(StudioPreferences.developerModeKey, store: StudioPreferences.defaults) private var developerMode = false

    var body: some View {
        VStack(spacing: 0) {
          HSplitView {
            Form {
                Section("Training data") {
                    HStack {
                        Button("Open Dataset…") { store.chooseDataset() }
                            .help("Choose an existing dataset folder containing your material maps.")
                        Button("Dataset Info…") { store.showDatasetInfoSheet = true }.disabled(store.dataset == nil)
                        Button("New Dataset…") { store.showNewDatasetSheet = true }
                            .help("Open Material Dataset to create or rename datasets, edit their info, and add or remove materials.")
                    }.disabled(store.isBusy)
                    if let dataset = store.dataset {
                        Text(store.datasetName).font(.headline).textSelection(.enabled)
                        Text(store.datasetDisplayURL?.path ?? dataset.datasetPath)
                            .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        Text("\(dataset.materials.count) materials · \(dataset.samples.count) material sets")
                        Text(store.datasetNativeSizeLabel).font(.caption).foregroundStyle(.secondary)
                    } else {
                        Text("Create a dataset in Manage Datasets, add paired material maps, then open its folder here.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    // Avoid Swift 6.3 IRGen's actor-isolated bound-method conversion.
                    Picker("Training dimensions", selection: Binding(get: { store.training.size }, set: { store.selectTrainingSize($0) })) {
                        ForEach(Array(Set(store.supportedTrainingSizes + [store.training.size])).sorted(), id: \.self) { size in
                            Text("\(size) × \(size)").tag(size)
                        }
                    }.disabled(store.isBusy || store.supportedTrainingSizes.isEmpty)
                    if let plans = store.dataset?.trainingPlans { DatasetPlanSummary(plans: plans) }
                    Text("Every diffuse and target map uses this exact grid. Matching sources are referenced directly; each crop is saved once in temporary training storage; smaller source sets are excluded.")
                        .font(.caption).foregroundStyle(.secondary)
                    Toggle("Train only selected material", isOn: $store.training.useSelectedMaterialOnly)
                    if store.training.useSelectedMaterialOnly {
                        Text(store.selectedMaterialName.map { "Selected material: \($0)" } ?? "Select a material in Dataset.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    if !store.datasetPreparationSummary.isEmpty {
                        Text(store.datasetPreparationSummary).font(.caption).foregroundStyle(.secondary)
                    }
                }.disabled(store.isBusy)
                Section("Refine a material model") {
                    TrainingModelNameField(name: $store.training.modelName, suggestedName: store.suggestedTrainingModelName)
                    Text("Shown in Saved Models and exported with the checkpoint. Clear the field to use the suggested name.")
                        .font(.caption).foregroundStyle(.secondary)
                    Picker("Map", selection: Binding(get: { store.training.target }, set: { store.selectTrainingTarget($0) })) {
                        Text("Displacement").tag("height")
                        Text("Roughness").tag("roughness")
                        Text("Normals").tag("normal")
                    }
                    Toggle("Start from selected checkpoint", isOn: $store.training.useWarmStart)
                    Text("Training scope: \(store.training.scope == "map-decoder" ? "Map decoder and output branch" : "Map output branch")")
                        .font(.caption).foregroundStyle(.secondary)
                    Button("Choose Starting Checkpoint…") { store.chooseResumeCheckpoint() }
                    if store.training.useWarmStart {
                        if let checkpoint = store.selectedCheckpoint {
                            Text("Starting model: \(checkpoint.title)").font(.caption).textSelection(.enabled)
                            Text(checkpoint.modelSummary)
                                .font(.caption).foregroundStyle(.secondary)
                        } else {
                            Text("Choose a starting checkpoint before training.").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                    NumericField("Updates per map", value: $store.training.updatesPerCrop, atLeast: 1, unit: "updates")
                    NumericField("Time limit", value: $store.training.maxMinutes, greaterThan: 0, unit: "minutes")
                    Text(developerMode ? "Exports a full fused safetensors checkpoint and a separate LoRA. Upload to Hugging Face from Saved Models." : "Exports a separate safetensors LoRA for refining your material base.")
                        .font(.caption).foregroundStyle(.secondary)
                    if let issue = store.trainingConfigurationIssue {
                        Text(issue).font(.caption).foregroundStyle(.secondary)
                    }
                }.disabled(store.isBusy)
                Section("Training settings") {
                    NumericField("Learning rate", value: $store.training.learningRate, greaterThan: 0)
                    Picker("Optimizer", selection: $store.training.optimizer) {
                        Text("AdamW").tag("adamw")
                        Text("Adam").tag("adam")
                    }
                    NumericField("Gradient accumulation steps", value: $store.training.gradientAccumulationSteps, atLeast: 1, unit: "maps / update")
                    Text("One complete map is processed at a time. Gradients are averaged before each optimizer update. Increasing accumulation does more work per update.")
                        .font(.caption).foregroundStyle(.secondary)
                    NumericField("Weight decay", value: $store.training.weightDecay, atLeast: 0)
                    Picker("Learning rate schedule", selection: $store.training.learningRateSchedule) {
                        Text("Constant").tag("constant")
                        Text("Cosine decay").tag("cosine")
                    }
                    if store.training.learningRateSchedule == "cosine" {
                        NumericField("Minimum learning rate ratio", value: $store.training.minimumLearningRateRatio, in: Double(Float.leastNonzeroMagnitude)...1)
                        Text("Cosine decay finishes at this fraction of the starting learning rate.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    NumericField("Warmup", value: $store.training.warmupUpdates, atLeast: 0, unit: "updates")
                    DisclosureGroup("Advanced training settings") {
                        NumericField("Optimizer beta 1", value: $store.training.optimizerBeta1, in: 0...Double(Float(1).nextDown))
                        NumericField("Optimizer beta 2", value: $store.training.optimizerBeta2, in: 0...Double(Float(1).nextDown))
                        NumericField("Optimizer epsilon", value: $store.training.optimizerEpsilon, greaterThan: 0)
                        NumericField("Maximum gradient norm", value: $store.training.maxGradientNorm, atLeast: 0)
                        Text("0 disables gradient clipping. Warm starts restore weights and start fresh optimizer moments.")
                            .font(.caption).foregroundStyle(.secondary)
                        NumericField("Random seed", value: $store.training.seed, atLeast: 0)
                        NumericField("LoRA rank", value: $store.training.loraRank, atLeast: 1)
                            .disabled(store.training.useWarmStart && store.selectedCheckpoint?.schema == "texture-studio-material-lora-v1")
                        NumericField("LoRA alpha", value: $store.training.loraAlpha, greaterThan: 0)
                            .disabled(store.training.useWarmStart && store.selectedCheckpoint?.schema == "texture-studio-material-lora-v1")
                        if store.training.useWarmStart && store.selectedCheckpoint?.schema == "texture-studio-material-lora-v1" {
                            Text("The starting LoRA retains its recorded rank and alpha. These controls apply to new adapters.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                        Text("Float32 training. The fixed objective is value L1 + 4 × multiscale detail L1; the selected LoRA layers learn while base weights stay frozen.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                }.disabled(store.isBusy)
                Section("Validation & checkpoints") {
                    if let validation = store.dataset?.validation {
                        Text(validation.enabled ? (validation.quickCount == 0 ? "Quick checks are off. The configured validation pool is checked when saving." : "Up to \(validation.quickCount) crops per quick check; the configured validation pool is checked when saving.") : "Validation is off in Dataset Info.")
                            .font(.caption).foregroundStyle(.secondary)
                    }
                    TrainingIntervalField("Quick check every", value: $store.training.validationEvery, unit: $store.training.validationUnit)
                    Text("0 turns off periodic quick checks.")
                        .font(.caption).foregroundStyle(.secondary)
                    TrainingIntervalField("Save checkpoint every", value: $store.training.checkpointEvery, unit: $store.training.checkpointUnit)
                    Text("An epoch is one complete pass through the training materials. A step is one optimizer update. Each interval uses its selected unit.")
                        .font(.caption).foregroundStyle(.secondary)
                    Text("0 saves on request and at final export. Saving checks the full validation pool when validation is enabled. Save Checkpoint Now finishes the current step, saves, and continues training.")
                        .font(.caption).foregroundStyle(.secondary)
                }.disabled(store.isBusy)
                if developerMode {
                    Section("Developer controls") {
                        Picker("Refinement scope", selection: $store.training.scope) {
                            Text("Map output branch").tag("final-map")
                            Text("Map decoder").tag("map-decoder")
                        }
                        Text("The output branch refines the selected map's RRDB layers. The map decoder also adapts its decoder and tail, adding backward work and time per step.")
                            .font(.caption).foregroundStyle(.secondary)
                        Toggle("Upload full checkpoint after training", isOn: $store.uploadAfterTraining)
                        Toggle("Public Hugging Face model", isOn: $store.uploadPublic)
                        TextField("Hugging Face repository (automatic when blank)", text: $store.uploadRepo)
                        if store.uploadAfterTraining {
                            Text(store.uploadAccount.map { account in
                                let repository = store.effectiveUploadRepo.isEmpty ? "an automatic repository under \(account)" : store.effectiveUploadRepo
                                return "After training: upload to \(repository) · \(store.uploadPublic ? "Public" : "Private")"
                            } ?? "After training: upload when a saved Hugging Face login is available. Set the repository and visibility above.")
                                .font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
                        }
                    }.disabled(store.isBusy)
                }
            }.formStyle(.grouped).frame(minWidth: 380, idealWidth: 440, maxWidth: 520)
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text(store.isTraining ? "Training progress" : "Operation log").font(.headline)
                    Spacer()
                    if store.isBusy {
                        ProgressView().controlSize(.small)
                    }
                }
                if let progress = store.trainingProgress { TrainingProgressView(progress: progress) }
                ScrollView {
                    Text(store.logText.isEmpty ? "Training and dataset preparation progress appears here." : store.logText)
                        .font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }.frame(maxHeight: .infinity)
                if !store.validationSummary.isEmpty { Text(store.validationSummary).font(.callout) }
                if !store.activity.isEmpty { Text(store.activity).font(.caption).foregroundStyle(.secondary) }
                if let error = store.error { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
                HStack {
                    if let output = store.lastOutputURL { Button("Show Run Folder") { NSWorkspace.shared.activateFileViewerSelecting([output]) } }
                    if let log = store.lastLogURL { Button("Show Log") { NSWorkspace.shared.activateFileViewerSelecting([log]) } }
                }
            }.padding(20).frame(minWidth: 420)
          }
          Divider()
          HStack(spacing: 12) {
              Button("Saved Models & Export…") { showCheckpoints = true }
              Spacer()
              if store.isBusy { WorkbenchStopButtons(store: store) }
              Button("Train Material", systemImage: "play.fill") { store.startTraining() }
                  .buttonStyle(.borderedProminent)
                  .disabled(store.isBusy || store.trainingConfigurationIssue != nil)
                  .accessibilityIdentifier("training.start")
          }
          .padding(.horizontal, 20).padding(.vertical, 12)
          .accessibilityElement(children: .contain)
          .accessibilityIdentifier("training.actions")
        }
        .sheet(isPresented: $showCheckpoints) { CheckpointLibraryView(store: store).frame(minWidth: 780, minHeight: 560) }
    }
}

struct TrainingIntervalField: View {
    let title: String
    @Binding var value: Int
    @Binding var unit: MaterialTrainingIntervalUnit

    init(_ title: String, value: Binding<Int>, unit: Binding<MaterialTrainingIntervalUnit>) {
        self.title = title
        _value = value
        _unit = unit
    }

    var body: some View {
        LabeledContent(title) {
            HStack(spacing: 8) {
                NumericTextField(title: title, value: $value, atLeast: 0)
                    .frame(width: 90)
                Picker("\(title) unit", selection: $unit) {
                    ForEach(MaterialTrainingIntervalUnit.allCases, id: \.self) { choice in
                        Text(choice.label).tag(choice)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(width: 78)
                .accessibilityLabel("\(title) unit")
            }
        }
    }
}

struct WorkbenchStopButtons: View {
    @Bindable var store: WorkbenchStore
    var body: some View {
        if store.isTraining {
            Button(store.isCheckpointPending ? "Checkpoint Queued…" : "Save Checkpoint Now") { store.saveCheckpointNow() }
                .disabled(!store.canStopAndSave || store.isCheckpointPending)
                .accessibilityIdentifier("training.save-checkpoint")
                .help("Run full validation when enabled, save a checkpoint, and continue training.")
            Button(store.isSavingTraining ? "Stopping & Saving…" : "Stop & Save", systemImage: "stop.fill") { store.stop() }
                .disabled(!store.canStopAndSave)
                .accessibilityIdentifier("training.stop")
                .help(store.hasTrainingStarted || store.isSavingTraining
                    ? "Finish the current update, validate when enabled, and save the material LoRA. Abort remains available while saving."
                    : "Available once training starts. Use Abort to cancel dataset preparation or model setup.")
            Button(role: .destructive) { store.abort() } label: {
                Label(store.isStopping && !store.isSavingTraining ? "Aborting…" : "Abort", systemImage: "xmark.octagon.fill")
            }
                .disabled(!store.canAbort)
                .accessibilityIdentifier("training.abort")
                .help("Cancel setup or training without a final save. Previously saved checkpoints are kept.")
        } else {
            Button(store.isStopping ? "Stopping…" : "Stop") { store.stop() }
                .disabled(!store.canAbort)
                .help("Cancel the current operation.")
        }
    }
}

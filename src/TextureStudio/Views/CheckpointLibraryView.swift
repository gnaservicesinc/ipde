import SwiftUI
import AppKit

struct CheckpointLibraryView: View {
    @Bindable var store: WorkbenchStore
    var showsDismissButton = true
    var onRefine: ((WorkbenchCheckpoint) -> Void)? = nil
    @AppStorage(StudioPreferences.developerModeKey, store: StudioPreferences.defaults) private var developerMode = false
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Saved Checkpoints").font(.title2.bold())
                    Text("Inspect matching saved outputs and export the exact checkpoint for another workflow.")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Spacer()
                if showsDismissButton { Button("Done") { dismiss() }.keyboardShortcut(.cancelAction) }
            }
            .padding(20)
            Divider()
            HSplitView {
                VStack(spacing: 0) {
                    if store.checkpoints.isEmpty {
                        ContentUnavailableView {
                            Label("Add a checkpoint", systemImage: "shippingbox")
                        } description: {
                            Text("Inspect a saved material checkpoint to view its target and training step.")
                        } actions: {
                            Button("Locate Checkpoints…") { store.chooseCheckpoint() }
                                .buttonStyle(.glassProminent)
                                .disabled(store.isBusy)
                        }
                    } else {
                        ScrollView {
                          LazyVStack(spacing: 4) {
                            ForEach(store.checkpoints) { checkpoint in
                              MaterialSidebarRow(selected: store.selectedCheckpointId == checkpoint.id, action: { store.selectedCheckpointId = checkpoint.id }) {
                                HStack(spacing: 9) {
                                    Image(systemName: "shippingbox").foregroundStyle(.secondary)
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(checkpoint.title).lineLimit(2)
                                        Text("\(targetTitle(checkpoint.target)) · step \(checkpoint.step)")
                                            .font(.caption).foregroundStyle(.secondary).lineLimit(1)
                                    }
                                }
                              }
                            }
                          }.padding(8)
                        }
                        .focusable().focusEffectDisabled()
                        .onKeyPress(.downArrow) { store.selectedCheckpointId = MaterialSidebarSelection.next(store.selectedCheckpointId, in: store.checkpoints.map(\.id), direction: 1); return .handled }
                        .onKeyPress(.upArrow) { store.selectedCheckpointId = MaterialSidebarSelection.next(store.selectedCheckpointId, in: store.checkpoints.map(\.id), direction: -1); return .handled }
                        .disabled(store.isBusy)
                    }
                    Divider()
                    VStack(alignment: .leading, spacing: 8) {
                        Button("Refresh My Hugging Face Models") { store.refreshHubModels() }.disabled(store.isBusy || store.uploadAccount == nil)
                        ForEach(store.hubModels) { model in
                            HStack {
                                Text(model.repository).font(.caption).lineLimit(2)
                                Spacer()
                                Button("Download") { store.downloadHubModel(model) }.disabled(store.isBusy)
                            }
                        }
                    }.padding(12)
                    HStack {
                        Button { store.chooseCheckpoint() } label: { Label("Locate…", systemImage: "plus") }
                            .disabled(store.isBusy)
                        Spacer()
                        Button { store.forgetSelectedCheckpoint() } label: { Label("Unlink", systemImage: "minus") }
                            .disabled(store.isBusy || store.selectedCheckpoint == nil)
                            .help("Remove the checkpoint from this library. Its file remains on disk.")
                    }
                    .padding(12)
                }
                .frame(minWidth: 260, idealWidth: 310, maxWidth: 390)
                if let checkpoint = store.selectedCheckpoint {
                    checkpointDetail(checkpoint)
                        .frame(minWidth: 390)
                } else {
                    ContentUnavailableView("Select a checkpoint", systemImage: "shippingbox", description: Text("Select a saved model to inspect or export it."))
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            }
            Divider()
            VStack(alignment: .leading, spacing: 6) {
                if store.isBusy {
                    HStack {
                        ProgressView().controlSize(.small)
                        Text(store.activity).font(.caption)
                    }
                } else if !store.activity.isEmpty {
                    Text(store.activity).font(.caption).foregroundStyle(.secondary)
                }
                if let error = store.error {
                    Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled)
                }
                Text("Unlinking keeps checkpoint files. Export creates a new package. Developer settings control uploads after training.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(14)
        }
        .task {
            // Workspace restoration may still be reconnecting checkpoints.
            // Account discovery is read-only and never uploads anything.
            while store.isBusy {
                do { try await Task.sleep(for: .milliseconds(100)) }
                catch { return }
            }
            if !store.uploadAccountChecked { store.refreshUploadAccount() }
        }
        .onChange(of: store.uploadRepo) { _, _ in store.saveUploadConfiguration() }
        .onChange(of: store.uploadPublic) { _, _ in store.saveUploadConfiguration() }
    }

    private func checkpointDetail(_ checkpoint: WorkbenchCheckpoint) -> some View {
        Form {
            Section("Selected model") {
                Text(checkpoint.title).font(.headline).textSelection(.enabled)
                LabeledContent("Target", value: targetTitle(checkpoint.target))
                LabeledContent(checkpoint.modelFamily?.isCompact == true ? "Model family" : "Training base",
                               value: checkpoint.modelFamily?.isCompact == true ? checkpoint.modelFamily?.label ?? checkpoint.trainingBaseLabel : checkpoint.trainingBaseLabel)
                LabeledContent("Availability", value: checkpoint.availabilityLabel)
                LabeledContent("Saved step", value: checkpoint.step.formatted())
                LabeledContent("Interface", value: checkpoint.compatible ? "Compatible" : "Unsupported")
                Text(checkpoint.schema).font(.caption).foregroundStyle(.secondary).textSelection(.enabled)
            }
            Section("Use this model") {
                Button("Use in Texture Studio") { store.useSelectedInStudio() }.disabled(store.isBusy || !checkpoint.supportsStudioInference)
                Button("Open Trainer with This Model") {
                    if store.selectCheckpointForRefinement(checkpoint, navigate: onRefine == nil) {
                        onRefine?(checkpoint)
                        if showsDismissButton { dismiss() }
                    }
                }
                    .disabled(store.isBusy || !checkpoint.supportsTrainingWarmStart)
                    .help("Open training with this checkpoint's map and refinement scope already selected.")
            }
            Section("Model package") {
                Text(checkpoint.modelFamily?.isCompact == true ? "Complete compact safetensors model" : developerMode ? "Full fused safetensors checkpoint + separate LoRA" : "Separate safetensors LoRA")
                    .font(.caption).foregroundStyle(.secondary)
                Button("Export Package…") { store.exportSelectedCheckpoint() }
                    .disabled(store.isBusy)
                    .help("Create a new portable package containing learned weights, a model card and dependency identities. Training photos and optimizer state stay local.")
                if store.lastPackageCheckpointId == checkpoint.id, let package = store.lastPackageURL {
                    LabeledContent("Export of this model", value: package.lastPathComponent)
                    Button("Show Package") { NSWorkspace.shared.activateFileViewerSelecting([package]) }
                }
            }
            Section("Upload to Hugging Face") {
                HStack(alignment: .firstTextBaseline) {
                    LabeledContent("Account", value: store.uploadAccount ?? "Not signed in")
                    Button("Refresh Account") { store.refreshUploadAccount() }.disabled(store.isBusy)
                        .help("Check the saved Hugging Face token in Apple Keychain. No files are uploaded.")
                }
                if store.uploadAccount == nil {
                    Text(store.uploadAccountMessage).font(.caption).foregroundStyle(.secondary)
                        .textSelection(.enabled)
                    Button("Copy Login Command") {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString("hf auth login", forType: .string)
                    }.help("Paste this command into Terminal to sign in, then click Refresh Account.")
                }
                TextField("Repository (automatic when blank)", text: $store.uploadRepo,
                          prompt: Text(store.suggestedUploadRepo.isEmpty ? "owner/model-name" : store.suggestedUploadRepo))
                    .help("Leave blank to create a separate repository for this selected model in your account, or enter owner/model-name to use a particular repository. This choice is saved.")
                    .disabled(store.isBusy)
                if !store.effectiveUploadRepo.isEmpty {
                    LabeledContent("Destination", value: store.effectiveUploadRepo).textSelection(.enabled)
                    if !HuggingFaceUpload.validRepository(store.effectiveUploadRepo) {
                        Text("Use owner/model-name with letters, numbers, underscores, dots or single hyphens.")
                            .font(.caption).foregroundStyle(.red)
                    }
                }
                Toggle("Public repository", isOn: $store.uploadPublic).disabled(store.isBusy)
                    .help("Off creates a private repository. Existing repository visibility must match this choice; the app never changes its visibility silently.")
                Text("Upload packages the selected checkpoint and its model card automatically. Source photos, optimizer state and base model files stay local.")
                    .font(.caption).foregroundStyle(.secondary)
                Button { store.uploadPackage() } label: {
                    Label("Upload Selected Model", systemImage: "square.and.arrow.up")
                }
                .buttonStyle(.glassProminent)
                .disabled(store.isBusy || !store.canUploadSelectedCheckpoint)
                .help("Upload this selected model directly to the destination shown above. A fresh, checksum-verified package is created first. No additional confirmation is shown.")
                if let uploaded = store.lastUploadURL {
                    Link("Open Last Uploaded Model", destination: uploaded)
                }
            }
            if developerMode, checkpoint.modelFamily == .pbrnxt {
                Section("Mix compatible LoRAs") {
                    Button("Add LoRA…") { store.chooseMixAdapter() }.disabled(store.isBusy)
                    ForEach($store.adapterMix) { $adapter in
                        HStack {
                            Text(URL(fileURLWithPath: adapter.path).lastPathComponent).lineLimit(1)
                            Text("Weight").foregroundStyle(.secondary)
                            NumericTextField(title: "LoRA weight for \(URL(fileURLWithPath: adapter.path).lastPathComponent)", value: $adapter.weight)
                                .frame(width: 100)
                            Button("Remove") { store.adapterMix.removeAll { $0.id == adapter.id } }
                        }
                    }
                    Text("Export verifies the exact base identity and target modules. Weights scale each adapter delta; image size alone does not establish compatibility.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            Section("File identity") {
                Text(checkpoint.checkpointPath).textSelection(.enabled)
                Text("SHA256: \(checkpoint.sha256)").textSelection(.enabled)
                Button("Show Checkpoint") { NSWorkspace.shared.activateFileViewerSelecting([checkpoint.url]) }
            }
            .font(.caption)
        }
        .formStyle(.grouped)
    }

    private func targetTitle(_ value: String) -> String {
        switch value {
        case "height": "Surface height / displacement"
        case "roughness": "Roughness"
        case "normal": "OpenGL Normal"
        default: value.capitalized
        }
    }
}

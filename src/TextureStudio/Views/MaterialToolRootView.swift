import SwiftUI

struct MaterialToolRootView: View {
    let role: MaterialTool
    let embeddedInHub: Bool
    @Bindable var store: WorkbenchStore
    @State private var review = ReviewSessionStore()
    @State private var showModels = false
    @State private var showRuntime = false
    init(role: MaterialTool, store: WorkbenchStore, review: ReviewSessionStore? = nil, embeddedInHub: Bool = false) {
        self.role = role
        self.embeddedInHub = embeddedInHub
        self.store = store
        self._review = State(initialValue: review ?? ReviewSessionStore())
    }
    private var comparisonReady: Bool {
        store.comparisonConfigurationIssue == nil
    }

    var body: some View {
        Group {
            switch role {
            case .review: reviewView
            case .compare: compareView
            case .dataset: DatasetWorkbenchView(store: store)
            case .train: TrainingWorkbenchView(store: store)
            }
        }
        .navigationTitle(role.title)
        .focusedSceneValue(\.materialDatasetStore, role == .dataset ? store : nil)
        .frame(minWidth: 1050, minHeight: 700)
        .toolbar {
            ToolbarItemGroup {
                Button { showModels = true } label: { Label("Checkpoints", systemImage: "shippingbox") }
                    .help("Inspect saved checkpoints, compare their outputs or export a portable package.")
                Button { showRuntime = true } label: { Label("Runtime", systemImage: "gearshape") }
                    .help("Configure the working folder and native model weights. Developer mode exposes material model controls.")
                Menu {
                    ForEach(MaterialTool.allCases) { tool in
                        Button(tool.title, systemImage: tool.symbol) { MaterialToolLauncher.open(tool) }
                    }
                    Button("Texture Studio", systemImage: "square.3.layers.3d") { MaterialToolLauncher.openStudio() }
                } label: { Label("Tools", systemImage: "macwindow.on.rectangle") }
            }
        }
        .sheet(isPresented: $showModels) {
            CheckpointLibraryView(store: store, onRefine: embeddedInHub || role == .train ? nil : { checkpoint in
                launchTrainer(checkpoint)
            }).frame(minWidth: 850, minHeight: 620)
        }
        .sheet(isPresented: $showRuntime) { WorkbenchRuntimeView(store: store).frame(width: 710, height: 500) }
        .modifier(DatasetManagementPresentation(store: store))
        .alert("Material tool", isPresented: Binding(get: { (store.error != nil || review.error != nil) && !store.showNewDatasetSheet && !store.showDatasetInfoSheet && !store.showAddMaterialSheet && !store.showImportFolderSheet }, set: { if !$0 { store.error = nil; review.error = nil } })) {
            Button("OK") { store.error = nil; review.error = nil }
        } message: { Text(store.error ?? review.error ?? "") }
        .task {
            store.restore()
            if role == .review && review.groups.isEmpty { review.restore(workspace: store.workspacePath) }
        }
        .onOpenURL { url in
            if url.lastPathComponent == MaterialTrainingHandoff.documentName {
                do {
                    let handoff = try MaterialTrainingHandoff.read(from: url)
                    store.receiveTrainingHandoff(handoff)
                    _ = try? handoff.discardTemporaryFile(at: url)
                } catch { store.error = error.localizedDescription }
                return
            }
            var isDirectory: ObjCBool = false
            let directoryExists = FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory) && isDirectory.boolValue
            let containsDataset = directoryExists && (FileManager.default.fileExists(atPath: url.appendingPathComponent("dataset.json").path)
                || (url.lastPathComponent == "sources" && FileManager.default.fileExists(atPath: url.deletingLastPathComponent().appendingPathComponent("dataset.json").path)))
            if url.lastPathComponent == "dataset.json" || containsDataset || (role == .dataset && directoryExists) { store.receiveDataset(url) }
            else if url.pathExtension == "json" { review.load(url) }
            else if url.pathExtension.lowercased() == "safetensors" {
                if role == .train { store.receiveResumeCheckpoint(url) }
                else { store.openCheckpoint(url) }
            }
            else { review.openMaps([url]) }
        }
    }

    private func launchTrainer(_ checkpoint: WorkbenchCheckpoint) {
        do {
            try MaterialToolLauncher.openTrainer(checkpoint: checkpoint, dataset: store.datasetDisplayURL,
                training: store.training, sampleID: store.selectedSampleId, inputVariantID: store.selectedInputVariantId,
                onFailure: { store.error = $0.localizedDescription })
        } catch { store.error = error.localizedDescription }
    }

    private var reviewView: some View {
        NavigationSplitView {
            ScrollView {
                LazyVStack(spacing: 4) {
                    ForEach(review.groups) { group in
                        MaterialSidebarRow(selected: review.selectedGroupId == group.id, action: { review.selectedGroupId = group.id }) {
                            Label(group.id.replacingOccurrences(of: "_", with: " "), systemImage: "square.3.layers.3d")
                        }
                    }
                }.padding(8)
            }.focusable().focusEffectDisabled()
                .onKeyPress(.downArrow) { review.selectedGroupId = MaterialSidebarSelection.next(review.selectedGroupId, in: review.groups.map(\.id), direction: 1); return .handled }
                .onKeyPress(.upArrow) { review.selectedGroupId = MaterialSidebarSelection.next(review.selectedGroupId, in: review.groups.map(\.id), direction: -1); return .handled }
                .navigationSplitViewColumnWidth(min: 180, ideal: 230)
        } detail: {
            if let selected = review.selected {
                VStack(spacing: 0) {
                    HStack {
                        Label("Sample: \(selected.id)", systemImage: "photo").font(.headline).textSelection(.enabled)
                        Spacer()
                    }.padding(.horizontal, 12).padding(.top, 12)
                    HStack {
                        let candidateId = selected.candidates.first(where: { $0.id == review.selectedCandidateId })?.id ?? selected.candidates.last!.id
                        Picker("Candidate", selection: Binding(get: { candidateId }, set: { review.selectedCandidateId = $0 })) {
                            ForEach(selected.candidates) { candidate in Text(candidate.label).tag(candidate.id) }
                        }.frame(minWidth: 260, idealWidth: 350)
                            .help("The decision and note apply to this named candidate for the sample shown above. Map pane titles identify each visible result.")
                        Picker("Decision", selection: Binding(get: { review.decisions[candidateId] ?? "unreviewed" }, set: { review.decisions[candidateId] = $0 })) {
                            Text("Unreviewed").tag("unreviewed")
                            Text("Usable").tag("usable")
                            Text("Needs work").tag("needs_work")
                            Text("Reject").tag("reject")
                        }.frame(width: 240)
                            .help("Record whether this candidate produces useful surface detail. A score alone cannot judge material quality.")
                        TextField("Detail, noise or relief observations", text: Binding(get: { review.notes[candidateId] ?? "" }, set: { review.notes[candidateId] = $0 }))
                        Button("Save Decisions & Notes…") { review.saveReview() }
                            .help("Save decisions and notes to a separate file you can reopen. Original maps and selected models stay intact.")
                    }.padding(12)
                    Divider()
                    ReviewWorkbenchView(candidates: selected.candidates, blendURL: review.blendURL,
                        onMissingSource: { url in
                            review.removeMissingSource(url)
                        }, onConfirmReview: { candidateID, recommendation in
                            review.decisions[candidateID] = recommendation == .approve ? "usable" : recommendation == .exclude ? "reject" : "needs_work"
                            review.selectedCandidateId = candidateID
                        })
                }
            } else {
                ContentUnavailableView("Inspect material details", systemImage: "photo.on.rectangle", description: Text("Open original maps or a review manifest. Each map supports native resolution, pan, zoom, pop-out and lossless export."))
            }
        }
        .toolbar {
            ToolbarItemGroup(placement: .navigation) {
                Button("Open Review…", systemImage: "folder") { review.chooseManifest() }
                    .help("Reopen saved review decisions or the review manifest produced by a checkpoint comparison.")
                Button("Open Maps…", systemImage: "photo") { review.chooseMaps() }
                    .help("Inspect original PNG or EXR maps at full source resolution. Export Original saves a map; Export Visible Maps saves all visible panes together.")
            }
        }
    }
    private var compareView: some View {
        VStack(spacing: 0) {
            HStack {
                Button("Test Photo…", systemImage: "photo") { store.chooseSourceImage() }
                    .disabled(store.isBusy)
                    .help("Use the same prepared diffuse at the chosen grid, for every checkpoint. A selected dataset material is used when no separate photo is chosen.")
                VStack(alignment: .leading, spacing: 3) {
                    Text(store.sourceImageURL.map { "Source photo: \($0.lastPathComponent)" }
                         ?? store.selectedSampleId.map { "Dataset material: \($0)" } ?? "Choose a photo or dataset material")
                        .lineLimit(1).truncationMode(.middle)
                    if store.sourceImageURL != nil {
                        Text("This saved photo is used instead of the dataset selection.").font(.caption)
                    }
                }.foregroundStyle(.secondary)
                if store.sourceImageURL != nil {
                    Button("Use Dataset Material") { store.sourceImageURL = nil }
                        .disabled(store.isBusy || store.selectedDiffuseMap == nil)
                        .help("Switch back to the material currently selected in Dataset. Its diffuse and reference map will be used for this comparison.")
                }
                Spacer()
                Button("Choose Checkpoints…") { store.chooseCheckpoint() }
                    .disabled(store.isBusy)
                    .help("Select one checkpoint to compare with its material base, or multiple saved checkpoints predicting the same map type. Their exact hashes are recorded with the results.")
                Button("Run Comparison", systemImage: "play.fill") { store.compare() }
                    .buttonStyle(.glassProminent).disabled(store.isBusy || !comparisonReady)
                    .help("Run checked candidates one at a time on Metal, then inspect their matching full-resolution outputs.")
            }.padding()
            if store.sourceImageURL != nil {
                DisclosureGroup("Prepare test diffuse") {
                    HStack(alignment: .top, spacing: 20) {
                        VStack {
                            DoubleControl(title: "Tilt X", value: $store.testPhotoSettings.rotationX, range: -70...70, suffix: "°")
                            DoubleControl(title: "Tilt Y", value: $store.testPhotoSettings.rotationY, range: -70...70, suffix: "°")
                            DoubleControl(title: "Rotate", value: $store.testPhotoSettings.rotationZ, range: -180...180, suffix: "°", enforcesSliderRange: false)
                        }
                        VStack {
                            FloatControl(title: "Lighting balance", value: $store.testPhotoSettings.lightingStrength, range: 0...1)
                            FloatControl(title: "Lighting scale", value: $store.testPhotoSettings.lightingRadius, range: 0...1)
                            Text("The shared Studio pipeline prepares one diffuse map at the selected grid. Inspect it alongside predictions before rating the result.")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }.padding(.horizontal).padding(.bottom, 10).disabled(store.isBusy)
            }
            ScrollView(.horizontal) {
                HStack {
                    ForEach(store.checkpoints) { checkpoint in
                        Toggle(isOn: Binding(get: { store.comparisonCheckpointIds.contains(checkpoint.id) }, set: { enabled in
                            if enabled { store.comparisonCheckpointIds.insert(checkpoint.id) } else { store.comparisonCheckpointIds.remove(checkpoint.id) }
                        })) { Text("\(checkpoint.title) · \(checkpoint.target) · step \(checkpoint.step)") }.toggleStyle(.checkbox)
                    }
                }.padding(.horizontal)
            }.disabled(store.isBusy)
            Toggle(store.comparisonBaselineLabel, isOn: $store.comparisonIncludesBase)
                .toggleStyle(.checkbox).disabled(store.isBusy).padding(.horizontal).padding(.vertical, 8)
            if let issue = store.comparisonConfigurationIssue, !store.isBusy {
                Text(issue)
                    .font(.caption).foregroundStyle(.secondary).padding(.horizontal).padding(.vertical, 8)
            }
            Divider()
            if !store.comparisonCandidates.isEmpty { ReviewWorkbenchView(candidates: store.comparisonCandidates) }
            else { ContentUnavailableView("Compare material models", systemImage: "rectangle.split.2x1", description: Text("Compare saved checkpoints on the same photo and inspect matching details. Each result identifies its source sample, training step and exact model. Source photos and dataset targets are available in Maps.")) }
            WorkbenchActivityView(store: store)
        }
    }
}

struct WorkbenchActivityView: View {
    @Bindable var store: WorkbenchStore
    var body: some View {
        HStack {
            if store.isBusy { ProgressView().controlSize(.small) }
            Text(store.activity).lineLimit(2).font(.caption)
            Spacer()
            if let url = store.lastOutputURL { Button("Show Results") { NSWorkspace.shared.activateFileViewerSelecting([url]) } }
            if store.isBusy { WorkbenchStopButtons(store: store) }
        }.padding(12)
    }
}

struct WorkbenchRuntimeView: View {
    @AppStorage(StudioPreferences.developerModeKey, store: StudioPreferences.defaults) private var developerMode = false
    @Bindable var store: WorkbenchStore
    @Environment(\.dismiss) private var dismiss
    @State private var hubToken = ""
    @State private var credentialMessage = ""
    var body: some View {
        VStack {
            Form {
                Section("Working folder") {
                    runtimeRow("Workspace", path: store.workspacePath, choose: store.chooseWorkspace)
                    Text("Choose a working folder for datasets, results and logs. Material processing and training use Apple frameworks on this Mac.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("Material base") {
                    runtimeRow("Weights", path: store.modelDirectory, choose: store.chooseEncoder)
                    HStack {
                        Button("Download Base Model") { store.installEncoder() }.disabled(store.isBusy)
                        Button("Remove Downloaded Base Weights", role: .destructive) { store.removeDownloadedEncoder() }.disabled(store.isBusy)
                    }
                    Text("Keep the base while refining a LoRA. A full fused checkpoint contains its weights; downloaded base weights can then be removed and obtained again.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("Hugging Face account") {
                    SecureField("Access token", text: $hubToken)
                    HStack {
                        Button("Save Token") {
                            do { try NativeHubCredentials.save(hubToken); hubToken = ""; credentialMessage = "Token saved in Keychain."; store.refreshUploadAccount() }
                            catch { credentialMessage = error.localizedDescription }
                        }.disabled(hubToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || store.isBusy)
                        Button("Sign Out") {
                            do { try NativeHubCredentials.save(""); credentialMessage = "Saved token removed."; store.refreshUploadAccount() }
                            catch { credentialMessage = error.localizedDescription }
                        }.disabled(store.isBusy)
                    }
                    Text(credentialMessage.isEmpty ? "Store a token with the permissions needed for your model repositories. Credentials are saved in Apple Keychain." : credentialMessage)
                        .font(.caption).foregroundStyle(.secondary)
                }
                Section("Settings") {
                    PreparationSettingsView(store: store)
                    Toggle("Developer mode", isOn: $developerMode)
                    Text("Expose adapter controls and export a full fused checkpoint alongside the separate LoRA.")
                        .font(.caption).foregroundStyle(.secondary)
                }

            }.formStyle(.grouped)
            Button("Done") { store.saveConfiguration(); dismiss() }.keyboardShortcut(.defaultAction).padding()
        }
    }
    private func runtimeRow(_ title: String, path: String, choose: @escaping () -> Void) -> some View {
        LabeledContent(title) { Text(path).lineLimit(2).font(.caption).textSelection(.enabled); Button("Locate…", action: choose) }
    }
}

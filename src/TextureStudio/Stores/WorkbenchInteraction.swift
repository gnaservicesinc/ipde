import Foundation

extension WorkbenchStore {
    /// Folder and crop selection always display their diffuse image first.
    func selectDatasetSample(_ sample: WorkbenchSample, role: String? = nil) {
        guard !isBusy else { return }
        if selectedSampleId != sample.id { selectedInputVariantId = nil }
        selectedSampleId = sample.id
        selectedRole = role.flatMap { sample.maps[$0] != nil ? $0 : nil }
            ?? ["input", "height", "roughness", "normal"].first(where: { sample.maps[$0] != nil }) ?? "input"
    }

    /// Both library and file-picker refinement use the checkpoint's recorded setup.
    @discardableResult
    func selectCheckpointForRefinement(_ checkpoint: WorkbenchCheckpoint, navigate: Bool = true) -> Bool {
        guard !isBusy else { return false }
        guard checkpoint.supportsTrainingWarmStart else {
            error = "This checkpoint does not support material refinement."
            return false
        }
        configureCheckpointForRefinement(checkpoint)
        if navigate { trainingNavigationRequest = UUID() }
        return true
    }

    func configureCheckpointForRefinement(_ checkpoint: WorkbenchCheckpoint) {
        if let family = checkpoint.modelFamily, family != training.modelFamily {
            if family.isCompact != training.modelFamily.isCompact { training.learningRate = family.defaultLearningRate }
            training.modelFamily = family
        }
        selectedCheckpointId = checkpoint.id
        training.target = checkpoint.target
        training.scope = checkpoint.modelFamily?.isCompact == true ? "full-model" : checkpoint.scope ?? "final-map"
        training.useWarmStart = true
    }

    func checkpointPackageArguments(for checkpoint: WorkbenchCheckpoint, output: URL, developer: Bool) -> [String] {
        var arguments = ["package", "--checkpoint", checkpoint.checkpointPath, "--expected-sha256", checkpoint.sha256,
                         "--output", output.path] + dependencyArguments(for: checkpoint)
        if developer { arguments += ["--developer-mode"] }
        if checkpoint.modelFamily == .pbrnxt {
            for adapter in adapterMix { arguments += ["--adapter", "\(adapter.path)=\(adapter.weight)"] }
        }
        return arguments
    }

    var comparisonConfigurationIssue: String? {
        if sourceImageURL == nil && selectedDiffuseMap == nil {
            return "Choose a test photo or select a dataset material with a diffuse map."
        }
        let selected = checkpoints.filter { comparisonCheckpointIds.contains($0.id) }
        if selected.isEmpty { return "Choose a checkpoint to compare with its base model." }
        if selected.contains(where: { !$0.supportsStudioInference }) {
            return "Select compatible material checkpoints for this comparison."
        }
        if Set(selected.map(\.target)).count != 1 {
            return "Select checkpoints for the same map: displacement, roughness or normals."
        }
        if selected.count == 1 && !comparisonIncludesBase {
            return "Include the base model or select a second checkpoint."
        }
        return nil
    }
}

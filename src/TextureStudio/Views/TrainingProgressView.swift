import SwiftUI

struct TrainingProgressView: View {
    let progress: WorkbenchTrainingProgress

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(alignment: .top, spacing: 8) {
                ForEach(Array(WorkbenchTrainingProgress.phaseLabels.enumerated()), id: \.offset) { index, label in
                    VStack(spacing: 4) {
                        Image(systemName: phaseFinished(index + 1) ? "checkmark.circle.fill" : "\(index + 1).circle\(progress.phaseIndex == index + 1 ? ".fill" : "")")
                            .font(.title3)
                        Text(label).font(.caption)
                    }
                    .foregroundStyle(progress.phaseIndex == index + 1 ? Color.accentColor : Color.secondary)
                    .frame(maxWidth: .infinity)
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("Phase \(index + 1) of \(progress.phaseCount): \(label)\(progress.phaseIndex == index + 1 ? ", current" : "")")
                }
            }
            Divider()
            HStack {
                Text(stepSummary)
                    .font(.headline).monospacedDigit()
                    .accessibilityIdentifier("training.total-steps")
                Spacer()
                if progress.elapsedSeconds > 0 {
                    Text(Duration.seconds(progress.elapsedSeconds).formatted(.time(pattern: .hourMinuteSecond)))
                        .font(.caption).monospacedDigit().foregroundStyle(.secondary)
                        .help("Elapsed time since the native model was ready, including validation and saving.")
                }
            }
            if progress.totalUpdates > 0 {
                ProgressView(value: progress.fractionCompleted)
                    .accessibilityLabel("Completed training steps")
            }
            if progress.skippedSampleCount > 0 {
                Text("Skipped samples: \(progress.skippedSampleCount.formatted()) · see operation log for details")
                    .font(.caption).foregroundStyle(.secondary)
                    .accessibilityIdentifier("training.skipped-samples")
            }
            HStack(spacing: 12) {
                if let update = progress.currentUpdateSummary { Text(update) }
                if let epoch = progress.epochSummary { Text(epoch) }
            }.font(.callout).monospacedDigit()
            Text(progress.operationLabel).font(.headline)
                .accessibilityIdentifier("training.current-operation")
            if !progress.operationDetail.isEmpty, progress.operationDetail != progress.operationLabel {
                Text(progress.operationDetail).font(.callout).foregroundStyle(.secondary)
            }
            if let sample = progress.sampleSummary {
                Text(sample + (progress.sampleID.map { " · \($0)" } ?? ""))
                    .font(.caption).textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
            if let stages = progress.stageSummary {
                HStack {
                    Text(stages).font(.caption).monospacedDigit()
                    ProgressView(value: Double(progress.stageCompleted), total: Double(progress.stageTotal))
                }
            }
            if let loss = progress.lastLoss {
                Text("Last step loss: \(loss.formatted(.number.precision(.significantDigits(4))))")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            HStack(spacing: 12) {
                if let seconds = progress.lastUpdateSeconds {
                    Text("Last step: \(Duration.seconds(seconds).formatted(.time(pattern: .minuteSecond)))")
                }
                if let rate = progress.learningRate {
                    Text("Learning rate: \(rate.formatted(.number.precision(.significantDigits(4))))")
                }
            }.font(.caption).foregroundStyle(.secondary).monospacedDigit()
            if progress.accumulationStep > 0, progress.gradientAccumulationSteps > 1 {
                Text("Accumulating map \(progress.accumulationStep) / \(progress.gradientAccumulationSteps) for this step")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            if progress.initialStep > 0 {
                Text("Checkpoint step \(progress.checkpointStep.formatted()) · started from \(progress.initialStep.formatted())")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
        }
        .padding(14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.quaternary, in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier("training.progress")
    }

    private var stepSummary: String {
        if progress.totalUpdates > 0 { return progress.updateSummary }
        if [.completed, .stopped, .aborted, .failed].contains(progress.state) {
            return "Total steps: \(progress.completedUpdates.formatted()) completed"
        }
        return "Total steps: counting prepared maps…"
    }

    private func phaseFinished(_ phase: Int) -> Bool {
        phase < progress.phaseIndex || (phase == 4 && [.completed, .stopped].contains(progress.state))
    }
}

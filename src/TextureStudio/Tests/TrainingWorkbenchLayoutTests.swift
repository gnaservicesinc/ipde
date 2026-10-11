import AppKit
import SwiftUI
import XCTest
@testable import TextureStudio

@MainActor
final class TrainingWorkbenchLayoutTests: XCTestCase {
    func testIntervalDropdownSitsToRightOfNumberAndPersistsUnitChanges() async throws {
        let suite = "training-interval-control-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults)
        let value = Binding(get: { store.training.validationEvery }, set: { store.training.validationEvery = $0 })
        let unit = Binding(get: { store.training.validationUnit }, set: { store.training.validationUnit = $0 })
        let host = NSHostingView(rootView: TrainingIntervalField("Quick check every", value: value, unit: unit).padding(20))
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 440, height: 120),
                              styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        window.orderFront(nil)
        defer { window.close() }
        try await waitFor(host) { Self.dropdown(in: host) != nil && Self.numberField(in: host) != nil }
        let field = try XCTUnwrap(Self.numberField(in: host))
        let dropdown = try XCTUnwrap(Self.dropdown(in: host))
        XCTAssertEqual(field.stringValue, "1")
        XCTAssertEqual(dropdown.itemTitles, ["epoch", "step"])
        XCTAssertEqual(dropdown.titleOfSelectedItem, "epoch")
        let numericFrame = field.convert(field.bounds, to: host)
        let unitFrame = dropdown.convert(dropdown.bounds, to: host)
        XCTAssertGreaterThan(unitFrame.minX, numericFrame.maxX, "Choose the interval unit immediately to the right of its input")
        dropdown.selectItem(withTitle: "step")
        dropdown.sendAction(dropdown.action, to: dropdown.target)
        try await waitFor(host) { store.training.validationUnit == .step }
        XCTAssertEqual(WorkbenchStore(preferences: defaults).training.validationUnit, .step)
        XCTAssertEqual(store.training.checkpointUnit, .epoch, "The two schedules have independent units")
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let attachment = XCTAttachment(image: NSImage(cgImage: try XCTUnwrap(bitmap.cgImage), size: host.bounds.size))
        attachment.name = "Training interval number and epoch / step dropdown"
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    func testTrainingAndSavingReserveBottomActionBarAtMinimumWindowSize() async throws {
        let suite = "training-layout-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults)
        store.operation("Training material", training: true) { try await Task.sleep(for: .seconds(60)) }
        defer { store.abort() }
        store.recordTrainingProgress("""
        {"event":"training_started","requested_updates":200,"updates_per_map":100,"initial_step":40}
        {"event":"update_started","completed_updates":12,"requested_updates":200,"current_update":13,"epoch":7,"total_epochs":100,"sample_position":1,"sample_total":2,"sample_id":"brick-center"}
        {"event":"operation_progress","phase":"training","operation":"Backward pass","completed":7,"total":18,"workflow_phase":3}

        """)
        let host = NSHostingView(rootView: TrainingWorkbenchView(store: store).environment(\.colorScheme, .dark))
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 1050, height: 700),
                              styleMask: [.titled, .closable, .resizable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: .darkAqua)
        window.contentView = host
        window.orderFront(nil)
        defer { window.close() }
        try await waitFor(host) { Self.splitView(in: host) != nil }
        try assertBottomActionBar(host)
        XCTAssertEqual(store.trainingProgress?.updateSummary, "Total steps: 12 / 200")
        XCTAssertEqual(store.trainingProgress?.currentUpdateSummary, "Running step 13 of 200")
        XCTAssertEqual(store.trainingProgress?.operationLabel, "Backward pass")

        store.saveCheckpointNow()
        store.stop()
        try await Task.sleep(for: .milliseconds(100))
        host.layoutSubtreeIfNeeded()
        try assertBottomActionBar(host)
        XCTAssertTrue(store.canAbort)
    }

    private func assertBottomActionBar(_ host: NSView) throws {
        let split = try XCTUnwrap(Self.splitView(in: host))
        let frame = split.convert(split.bounds, to: host)
        let bottomSpace = host.isFlipped ? host.bounds.maxY - frame.maxY : frame.minY - host.bounds.minY
        XCTAssertGreaterThanOrEqual(bottomSpace, 44, "Settings and the log must leave visible space for the bottom run controls")
        XCTAssertLessThanOrEqual(bottomSpace, 80, "The action bar must not consume the training area")
        XCTAssertEqual(frame.width, host.bounds.width, accuracy: 1)
    }

    private func waitFor(_ host: NSView, condition: () -> Bool) async throws {
        for _ in 0..<200 {
            host.layoutSubtreeIfNeeded()
            if condition() { return }
            try await Task.sleep(for: .milliseconds(10))
        }
        XCTFail("Training controls did not appear")
    }

    private static func splitView(in view: NSView) -> NSSplitView? {
        if let split = view as? NSSplitView { return split }
        return view.subviews.lazy.compactMap { splitView(in: $0) }.first
    }

    private static func dropdown(in view: NSView) -> NSPopUpButton? {
        if let dropdown = view as? NSPopUpButton { return dropdown }
        return view.subviews.lazy.compactMap { dropdown(in: $0) }.first
    }

    private static func numberField(in view: NSView) -> NSTextField? {
        if let field = view as? NSTextField, field.isEditable { return field }
        return view.subviews.lazy.compactMap { numberField(in: $0) }.first
    }
}

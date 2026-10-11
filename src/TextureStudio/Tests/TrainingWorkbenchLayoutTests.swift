import AppKit
import SwiftUI
import XCTest
@testable import TextureStudio

@MainActor
final class TrainingWorkbenchLayoutTests: XCTestCase {
    func testIntervalDropdownLayoutAndRenderedSelectionPersist() async throws {
        let suite = "training-interval-control-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults)
        let value = Binding(get: { store.training.validationEvery }, set: { store.training.validationEvery = $0 })
        let unit = Binding(get: { store.training.validationUnit }, set: { store.training.validationUnit = $0 })
        let host = NSHostingView(rootView: Form {
            Section("Validation & checkpoints") {
                TrainingIntervalField("Quick check every", value: value, unit: unit)
            }
        }.formStyle(.grouped).environment(\.colorScheme, .dark))
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 440, height: 160),
                              styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.appearance = NSAppearance(named: .darkAqua)
        window.contentView = host
        let previousKeyWindow = NSApp.keyWindow
        window.makeKeyAndOrderFront(nil)
        defer { window.close(); previousKeyWindow?.makeKey() }
        try await waitFor(host) { Self.numberField(in: host) != nil }
        window.recalculateKeyViewLoop()
        host.displayIfNeeded()
        try snapshot(host, name: "Training interval before querying dropdown")
        let field = try XCTUnwrap(Self.numberField(in: host))
        let control = Self.menuAnchorToRight(of: field, in: host)
        if control == nil { print("TRAINING_INTERVAL_NATIVE_HIERARCHY\n" + Self.nativeHierarchy(host)) }
        let dropdown = try XCTUnwrap(control, "The rendered unit menu must have its hosted view anchor")
        XCTAssertEqual(field.stringValue, "0")
        XCTAssertEqual(store.training.validationUnit, .epoch)
        let numericFrame = field.convert(field.bounds, to: host)
        let unitFrame = dropdown.convert(dropdown.bounds, to: host)
        XCTAssertGreaterThan(unitFrame.minX, numericFrame.maxX, "Choose the interval unit immediately to the right of its input")
        host.displayIfNeeded()
        let epochLabel = try renderedLabel(host, frame: unitFrame)
        // SwiftUI draws the menu itself and exposes a keyboard proxy in this
        // hosted XCTest environment. Verify the real rendered label changes
        // with its binding, without assuming a native popup or virtual AX tree.
        unit.wrappedValue = .step
        var stepLabel = epochLabel
        for _ in 0..<100 {
            try await Task.sleep(for: .milliseconds(10))
            host.layoutSubtreeIfNeeded(); host.displayIfNeeded()
            stepLabel = try renderedLabel(host, frame: unitFrame)
            if stepLabel != epochLabel { break }
        }
        XCTAssertNotEqual(stepLabel, epochLabel, "Changing epoch to step must update the visible menu label")
        XCTAssertEqual(WorkbenchStore(preferences: defaults).training.validationUnit, .step)
        XCTAssertEqual(store.training.checkpointUnit, .epoch, "The two schedules have independent units")
        try snapshot(host, name: "Training interval number and epoch / step dropdown")
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

    private static func menuAnchorToRight(of field: NSTextField, in host: NSView) -> NSView? {
        let numericFrame = field.convert(field.bounds, to: host)
        func isUnitControl(_ candidate: NSView) -> Bool {
            let frame = candidate.convert(candidate.bounds, to: host)
            // The diagnostic screenshot and native hierarchy identify this as
            // the menu's anchor on the current macOS SwiftUI implementation.
            // It is a view, rather than an NSPopUpButton or in-process AX node.
            return String(describing: type(of: candidate)) == "KeyViewProxy" && !candidate.isHidden &&
                frame.minX > numericFrame.maxX && frame.height >= 16 && frame.width >= 32
        }
        if let next = field.nextValidKeyView, isUnitControl(next) { return next }
        func find(_ view: NSView) -> NSView? {
            if isUnitControl(view) { return view }
            return view.subviews.lazy.compactMap { find($0) }.first
        }
        return find(host)
    }

    private func renderedLabel(_ host: NSView, frame: CGRect) throws -> Data {
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        // Crop the text portion, excluding the chevron and focus perimeter.
        let label = CGRect(x: frame.minX + 4, y: frame.minY + 3,
                           width: max(1, frame.width - 28), height: max(1, frame.height - 6))
        let scaleX = CGFloat(bitmap.pixelsWide) / host.bounds.width
        let scaleY = CGFloat(bitmap.pixelsHigh) / host.bounds.height
        let top = host.isFlipped ? label.minY : host.bounds.height - label.maxY
        let pixels = CGRect(x: label.minX * scaleX, y: top * scaleY,
                            width: label.width * scaleX, height: label.height * scaleY).integral
        let crop = try XCTUnwrap(bitmap.cgImage?.cropping(to: pixels))
        return try XCTUnwrap(NSBitmapImageRep(cgImage: crop).representation(using: .png, properties: [:]))
    }

    private func snapshot(_ host: NSView, name: String) throws {
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        let attachment = XCTAttachment(image: NSImage(cgImage: try XCTUnwrap(bitmap.cgImage), size: host.bounds.size))
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
        if let bytes = bitmap.representation(using: .png, properties: [:]) {
            let path = FileManager.default.temporaryDirectory.appendingPathComponent("ipde-training-interval-control.png")
            try bytes.write(to: path)
            print("TRAINING_INTERVAL_SNAPSHOT=\(path.path)")
        }
    }

    private static func nativeHierarchy(_ view: NSView, indent: String = "") -> String {
        let line = "\(indent)\(String(describing: type(of: view))) frame=\(view.frame) hidden=\(view.isHidden)"
        return ([line] + view.subviews.map { nativeHierarchy($0, indent: indent + "  ") }).joined(separator: "\n")
    }

    private static func numberField(in view: NSView) -> NSTextField? {
        if let field = view as? NSTextField, field.isEditable { return field }
        return view.subviews.lazy.compactMap { numberField(in: $0) }.first
    }
}

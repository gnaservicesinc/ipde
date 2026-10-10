import AppKit
import SwiftUI
import XCTest
@testable import TextureStudio

@MainActor
final class TrainingModelNameFieldTests: XCTestCase {
    func testFocusingSuggestedNameKeepsItVisibleAndAllowsEditing() async throws {
        try await checkEditing(initialName: "", suggestion: "Stone Displacement", expected: "Stone Displacement")
    }

    func testFocusingStoredNameKeepsItAndEditsPersistImmediately() async throws {
        try await checkEditing(initialName: "石 Stone / Warm evening", suggestion: "Automatic", expected: "石 Stone / Warm evening")
    }

    private func checkEditing(initialName: String, suggestion: String, expected: String) async throws {
        let suite = "training-name-control-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        let store = WorkbenchStore(preferences: defaults)
        store.training.modelName = initialName
        let name = Binding(get: { store.training.modelName }, set: { store.training.modelName = $0 })
        let host = NSHostingView(rootView: TrainingModelNameField(name: name, suggestedName: suggestion).padding(20))
        let window = NSWindow(contentRect: CGRect(x: 0, y: 0, width: 500, height: 140),
                              styleMask: [.titled, .closable], backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host
        window.orderFront(nil)
        defer { window.close() }
        var field: NSTextField?
        for _ in 0..<100 {
            host.layoutSubtreeIfNeeded()
            field = Self.textField(in: host)
            if field != nil { break }
            try await Task.sleep(for: .milliseconds(10))
        }
        let control = try XCTUnwrap(field)
        XCTAssertEqual(control.stringValue, expected)
        XCTAssertTrue(window.makeFirstResponder(control))
        try await Task.sleep(for: .milliseconds(50))
        let editor = try XCTUnwrap(control.currentEditor() as? NSTextView)
        XCTAssertEqual(editor.string, expected, "Focusing must preserve the displayed suggested or stored name")
        XCTAssertEqual(store.training.modelName, initialName, "Focus alone does not overwrite naming preferences")
        editor.setSelectedRange(NSRange(location: 0, length: editor.string.utf16.count))
        editor.insertText("Refined stone", replacementRange: editor.selectedRange())
        try await Task.sleep(for: .milliseconds(50))
        XCTAssertEqual(store.training.modelName, "Refined stone")
        XCTAssertEqual(WorkbenchStore(preferences: defaults).training.modelName, "Refined stone")
    }

    private static func textField(in view: NSView) -> NSTextField? {
        if let field = view as? NSTextField, field.isEditable { return field }
        return view.subviews.lazy.compactMap { textField(in: $0) }.first
    }
}

import SwiftUI

/// Render the automatic name as editable text. A placeholder disappears when
/// macOS focuses the editor, which made the suggested name look as if it was lost.
struct TrainingModelNameField: View {
    @Binding var name: String
    let suggestedName: String
    @State private var text: String
    @State private var lastAcceptedName: String
    @FocusState private var isFocused: Bool

    init(name: Binding<String>, suggestedName: String) {
        _name = name
        self.suggestedName = suggestedName
        _text = State(initialValue: Self.displayName(name.wrappedValue, suggestedName: suggestedName))
        _lastAcceptedName = State(initialValue: name.wrappedValue)
    }

    var body: some View {
        TextField("Model name", text: Binding(get: { text }, set: { value in
            // AppKit can echo the displayed string when creating its field
            // editor. Only an actual edit should pin an automatic name.
            guard value != text else { return }
            text = value
            lastAcceptedName = value
            name = value
        }))
        .accessibilityIdentifier("training.model-name")
        .focused($isFocused)
        .onChange(of: name) {
            if name != lastAcceptedName {
                text = Self.displayName(name, suggestedName: suggestedName)
                lastAcceptedName = name
            }
        }
        .onChange(of: suggestedName) {
            if !isFocused && name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                text = suggestedName
            }
        }
        .onChange(of: isFocused) {
            if !isFocused { text = Self.displayName(name, suggestedName: suggestedName) }
        }
    }

    private static func displayName(_ name: String, suggestedName: String) -> String {
        name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? suggestedName : name
    }
}

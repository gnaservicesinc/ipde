import SwiftUI

/// Numeric form fields keep a text draft so clearing a field or typing a sign
/// never fights the editor. Valid values are applied as they are typed.
protocol NumericEditableValue: Comparable, LosslessStringConvertible, Sendable {
    var isFiniteNumber: Bool { get }
}

extension Int: NumericEditableValue { var isFiniteNumber: Bool { true } }
extension UInt64: NumericEditableValue { var isFiniteNumber: Bool { true } }
extension Float: NumericEditableValue { var isFiniteNumber: Bool { isFinite } }
extension Double: NumericEditableValue { var isFiniteNumber: Bool { isFinite } }

enum NumericTextEditing {
    static func value<Value: NumericEditableValue>(
        from text: String,
        in range: ClosedRange<Value>? = nil,
        atLeast lowerBound: Value? = nil,
        greaterThan minimum: Value? = nil,
        decimalSeparator: String = Locale.current.decimalSeparator ?? "."
    ) -> Value? {
        var text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if decimalSeparator != "." { text = text.replacingOccurrences(of: decimalSeparator, with: ".") }
        guard let value = Value(text), value.isFiniteNumber,
              range?.contains(value) != false,
              lowerBound.map({ value >= $0 }) != false,
              minimum.map({ value > $0 }) != false else { return nil }
        return value
    }

    static func requirement<Value: NumericEditableValue>(
        in range: ClosedRange<Value>?, atLeast lowerBound: Value?, greaterThan minimum: Value?
    ) -> String {
        if let range { return "Enter a value from \(range.lowerBound) to \(range.upperBound)." }
        if let minimum { return "Enter a value greater than \(minimum)." }
        if let lowerBound { return "Enter a value of \(lowerBound) or greater." }
        return "Enter a finite number."
    }
}

/// A binding can also change through a slider, Reset, or another view. Preserve
/// the spelling of our own edits, but always show a value accepted elsewhere.
struct NumericTextDraft<Value: NumericEditableValue> {
    var text: String
    private(set) var lastAcceptedValue: Value?

    init(value: Value?) {
        text = value.map { String($0) } ?? ""
        lastAcceptedValue = value
    }

    mutating func accept(_ value: Value?) { lastAcceptedValue = value }

    mutating func synchronize(with value: Value?, isFocused: Bool) {
        guard !isFocused || value != lastAcceptedValue else { return }
        text = value.map { String($0) } ?? ""
        lastAcceptedValue = value
    }
}

struct NumericField<Value: NumericEditableValue>: View {
    private let title: String
    @Binding private var value: Value
    private let range: ClosedRange<Value>?
    private let lowerBound: Value?
    private let minimum: Value?
    private let unit: String

    init(_ title: String, value: Binding<Value>, in range: ClosedRange<Value>? = nil,
         atLeast lowerBound: Value? = nil, greaterThan minimum: Value? = nil, unit: String = "") {
        self.title = title
        _value = value
        self.range = range
        self.lowerBound = lowerBound
        self.minimum = minimum
        self.unit = unit
    }

    var body: some View {
        LabeledContent(title) {
            NumericTextField(title: title, value: $value, in: range, atLeast: lowerBound, greaterThan: minimum)
                .frame(width: 120)
            if !unit.isEmpty { Text(unit).foregroundStyle(.secondary) }
        }
    }
}

struct NumericTextField<Value: NumericEditableValue>: View {
    let title: String
    @Binding var value: Value
    var range: ClosedRange<Value>?
    var lowerBound: Value?
    var minimum: Value?

    init(title: String, value: Binding<Value>, in range: ClosedRange<Value>? = nil,
         atLeast lowerBound: Value? = nil, greaterThan minimum: Value? = nil) {
        self.title = title
        _value = value
        self.range = range
        self.lowerBound = lowerBound
        self.minimum = minimum
    }

    var body: some View {
        NumericDraftField(title: title, value: Binding(get: { value }, set: { if let updated = $0 { value = updated } }),
                          range: range, lowerBound: lowerBound, minimum: minimum, allowsEmpty: false)
    }
}

struct OptionalNumericTextField<Value: NumericEditableValue>: View {
    let title: String
    @Binding var value: Value?
    var minimum: Value?

    init(title: String, value: Binding<Value?>, greaterThan minimum: Value? = nil) {
        self.title = title
        _value = value
        self.minimum = minimum
    }

    var body: some View {
        NumericDraftField(title: title, value: $value, range: nil, lowerBound: nil, minimum: minimum, allowsEmpty: true)
    }
}

private struct NumericDraftField<Value: NumericEditableValue>: View {
    let title: String
    @Binding var value: Value?
    let range: ClosedRange<Value>?
    let lowerBound: Value?
    let minimum: Value?
    let allowsEmpty: Bool
    @State private var draft: NumericTextDraft<Value>
    @State private var message: String?
    @FocusState private var isFocused: Bool

    init(title: String, value: Binding<Value?>, range: ClosedRange<Value>?, lowerBound: Value?, minimum: Value?, allowsEmpty: Bool) {
        self.title = title
        _value = value
        self.range = range
        self.lowerBound = lowerBound
        self.minimum = minimum
        self.allowsEmpty = allowsEmpty
        _draft = State(initialValue: NumericTextDraft(value: value.wrappedValue))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            TextField(title, text: Binding(get: { draft.text }, set: { updateDraft($0) }),
                      prompt: allowsEmpty ? Text("Automatic") : nil)
                .textFieldStyle(.roundedBorder)
                .multilineTextAlignment(.trailing)
                .labelsHidden()
                .accessibilityLabel(title)
                .focused($isFocused)
                .onSubmit { finishEditing() }
            if let message { Text(message).font(.caption2).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true) }
        }
        .help(NumericTextEditing.requirement(in: range, atLeast: lowerBound, greaterThan: minimum) +
              (allowsEmpty ? " Leave blank to use the camera information." : " You can type a value directly."))
        .onChange(of: value) {
            draft.synchronize(with: value, isFocused: isFocused)
        }
        .onChange(of: isFocused) { if !isFocused { finishEditing() } }
    }

    private func updateDraft(_ text: String) {
        draft.text = text
        if allowsEmpty && text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            draft.accept(nil)
            value = nil
            message = nil
        } else if let updated: Value = NumericTextEditing.value(from: text, in: range, atLeast: lowerBound, greaterThan: minimum) {
            draft.accept(updated)
            value = updated
            message = nil
        }
    }

    private func finishEditing() {
        if allowsEmpty && draft.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            draft.accept(nil)
            value = nil
            message = nil
        } else if let updated: Value = NumericTextEditing.value(from: draft.text, in: range, atLeast: lowerBound, greaterThan: minimum) {
            draft.accept(updated)
            value = updated
            message = nil
        } else {
            message = NumericTextEditing.requirement(in: range, atLeast: lowerBound, greaterThan: minimum)
        }
        draft.synchronize(with: value, isFocused: false)
    }
}

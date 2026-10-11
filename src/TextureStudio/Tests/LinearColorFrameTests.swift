import CoreImage
import XCTest
@testable import TextureStudio

final class LinearColorFrameTests: XCTestCase {
    private let linear = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!

    func testFrozenFrameKeepsFloatRangeChannelsAndRowOrder() throws {
        let values: [Float] = [0.125, 0.25, 1.5, 1, 0.5, 0.75, 2.5, 1,
                               3.5, 0.0625, 0.875, 1, 4.5, 0.375, 0.625, 1]
        let original = values.withUnsafeBytes { Data($0) }
        let source = CIImage(bitmapData: original, bytesPerRow: 32, size: CGSize(width: 2, height: 2),
            format: .RGBAf, colorSpace: linear).transformed(by: CGAffineTransform(translationX: 7, y: 11))
        let context = CIContext(options: [.workingFormat: CIFormat.RGBAf, .workingColorSpace: linear])
        let frame = try LinearColorFrame(source, context: context)
        XCTAssertEqual(frame.color.extent, source.extent)
        XCTAssertEqual(frame.numeric.extent, source.extent)
        var restored = [Float](repeating: 0, count: values.count)
        let numeric = CIContext(options: [.workingFormat: CIFormat.RGBAf,
            .workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        restored.withUnsafeMutableBytes {
            numeric.render(frame.numeric, toBitmap: $0.baseAddress!, rowBytes: 32,
                bounds: source.extent, format: .RGBAf, colorSpace: nil)
        }
        for index in values.indices { XCTAssertEqual(restored[index], values[index], accuracy: 0.00001) }
        XCTAssertEqual(values.withUnsafeBytes { Data($0) }, original)
    }

    func testExposureAnchorHandlesBothHDRBoostAndLowerExposure() {
        let reference = image([0.2, 0.3, 0.4, 1])
        let context = CIContext(options: [.workingFormat: CIFormat.RGBAf, .workingColorSpace: linear])
        for factor: Float in [0.25, 8] {
            let hdr = image([0.2 * factor, 0.3 * factor, 0.4 * factor, 1])
            XCTAssertEqual(LinearColorFrame.exposureGain(reference: reference, hdr: hdr, context: context),
                1 / factor, accuracy: 0.0001)
        }
    }

    func testExposureAnchorDoesNotDivideBlackAndSupportsAllDarkSurfaces() {
        let context = CIContext(options: [.workingFormat: CIFormat.RGBAf, .workingColorSpace: linear])
        XCTAssertEqual(LinearColorFrame.exposureGain(reference: image([0, 0, 0, 1]),
            hdr: image([0, 0, 0, 1]), context: context), 1)
        XCTAssertEqual(LinearColorFrame.exposureGain(reference: image([0.005, 0.005, 0.005, 1]),
            hdr: image([0.04, 0.04, 0.04, 1]), context: context), 0.125, accuracy: 0.0001)
    }

    func testExposureAnchorUsesVisibleStraightColorAndExcludesZeroAlpha() {
        let context = CIContext(options: [.workingFormat: CIFormat.RGBAf, .workingColorSpace: linear])
        XCTAssertEqual(LinearColorFrame.exposureGain(reference: image([0.05, 0.075, 0.1, 0.25]),
            hdr: image([0.8, 1.2, 1.6, 0.5]), context: context), 0.125, accuracy: 0.0001,
                       "Different source alpha values do not change HDR exposure matching")
        XCTAssertEqual(LinearColorFrame.exposureGain(reference: image([0, 0, 0, 0]),
            hdr: image([0, 0, 0, 0]), context: context), 1)
    }

    private func image(_ rgba: [Float]) -> CIImage {
        CIImage(bitmapData: rgba.withUnsafeBytes { Data($0) }, bytesPerRow: 16,
            size: CGSize(width: 1, height: 1), format: .RGBAf, colorSpace: linear)
            .clampedToExtent().cropped(to: CGRect(x: 0, y: 0, width: 32, height: 32))
    }
}

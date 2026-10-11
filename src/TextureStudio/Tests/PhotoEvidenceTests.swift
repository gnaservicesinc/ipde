import XCTest
import CoreImage
@testable import TextureStudio

final class PhotoEvidenceTests: XCTestCase {
    func testExposureAndColorDifferencesAreCompensated() throws {
        let reference = texturedImage(seed: 11, gains: [1,1,1])
        let companion = texturedImage(seed: 11, gains: [0.6,0.8,0.7])
        let result = try PhotoEvidenceService.assess(reference: reference, companion: companion, context: CIContext())
        XCTAssertGreaterThan(result.acceptedFraction, 0.4)
        XCTAssertEqual(result.gains[0], 1/0.6, accuracy: 0.02)
        XCTAssertEqual(result.gains[1], 1/0.8, accuracy: 0.02)
        XCTAssertEqual(result.gains[2], 1/0.7, accuracy: 0.02)
    }

    func testUnrelatedSurfaceCannotContribute() throws {
        let result = try PhotoEvidenceService.assess(reference: texturedImage(seed: 11, gains: [1,1,1]),
            companion: texturedImage(seed: 971, gains: [1,1,1]), context: CIContext())
        XCTAssertLessThan(result.acceptedFraction, 0.01)
    }

    func testTexturelessViewDoesNotInventRegistrationEvidence() throws {
        let plain = CIImage(color: CIColor(red: 0.4, green: 0.4, blue: 0.4)).cropped(to: CGRect(x: 0,y: 0,width: 192,height: 192))
        let result = try PhotoEvidenceService.assess(reference: plain, companion: plain, context: CIContext())
        XCTAssertEqual(result.acceptedFraction, 0)
    }

    func testTransparentCompanionHoleNeverReceivesBlendWeight() throws {
        let reference = texturedImage(seed: 11, gains: [1,1,1])
        let companion = texturedImage(seed: 11, gains: [1,1,1], transparentPixel: (5,5))
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull()])
        let result = try PhotoEvidenceService.assess(reference: reference, companion: companion, context: context)
        XCTAssertGreaterThan(result.acceptedFraction, 0.4)
        var values = [Float](repeating: 0, count: 192*192)
        values.withUnsafeMutableBytes {
            context.render(result.mask, toBitmap: $0.baseAddress!, rowBytes: 192*4,
                           bounds: CGRect(x: 0,y: 0,width: 192,height: 192), format: .Rf, colorSpace: nil)
        }
        XCTAssertEqual(values[5*192+5], 0)
        XCTAssertEqual(values[5*192+6], 0.3, accuracy: 1e-6)
    }

    func testNativeFusionPreservesBothImagesTransparencyBetweenAssessmentSamples() {
        let referenceAlpha: [Float] = [1, 0, 0.4, 0.999, 1, 1, 1, 1]
        let companionAlpha: [Float] = [1, 1, 1, 1, 0, 0.6, 0.999, 1]
        let referenceColor: [Float] = [0.2, 0.3, 0.4], companionColor: [Float] = [0.6, 0.7, 0.8]
        let extent = CGRect(x: 7, y: 11, width: referenceAlpha.count, height: 1)
        func image(_ color: [Float], _ alpha: [Float]) -> CIImage {
            let rgba = alpha.flatMap { opacity in color.map { $0 * opacity } + [opacity] }
            return CIImage(bitmapData: rgba.withUnsafeBytes { Data($0) }, bytesPerRow: alpha.count * 16,
                           size: extent.size, format: .RGBAf, colorSpace: nil)
                .transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY))
        }
        let reference = image(referenceColor, referenceAlpha), companion = image(companionColor, companionAlpha)
        // One accepted assessment sample spans eight native pixels, including
        // holes and partial alpha that the coarse grid never represented.
        let mask = CIImage(color: CIColor(red: 0.3, green: 0.3, blue: 0.3)).cropped(to: CGRect(x: 0, y: 0, width: 1, height: 1))
        let blended = PhotoEvidenceService.blendRegisteredCompanion(reference: reference, companion: companion, assessmentMask: mask)
        XCTAssertEqual(blended.extent, extent)
        let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull(), .workingFormat: CIFormat.RGBAf])
        var actual = [Float](repeating: 0, count: referenceAlpha.count * 4)
        actual.withUnsafeMutableBytes {
            context.render(blended, toBitmap: $0.baseAddress!, rowBytes: referenceAlpha.count * 16,
                           bounds: extent, format: .RGBAf, colorSpace: nil)
        }
        for pixel in referenceAlpha.indices {
            XCTAssertEqual(actual[pixel * 4 + 3], referenceAlpha[pixel], accuracy: 1e-6,
                           "Supporting evidence must preserve native primary opacity")
            for channel in 0..<3 {
                let expected = referenceAlpha[pixel] == 1 && companionAlpha[pixel] == 1
                    ? referenceColor[channel] * 0.7 + companionColor[channel] * 0.3
                    : referenceColor[channel] * referenceAlpha[pixel]
                XCTAssertEqual(actual[pixel * 4 + channel], expected, accuracy: 1e-6,
                               "Only fully opaque native pixels may receive companion color")
            }
        }
    }

    private func texturedImage(seed: UInt32, gains: [Float], transparentPixel: (Int,Int)? = nil) -> CIImage {
        var state = seed
        var samples = [Float](repeating: 1, count: 192*192*4)
        for p in stride(from: 0, to: samples.count, by: 4) {
            state = state &* 1664525 &+ 1013904223
            let value = Float(state % 1000) / 2000 + 0.12
            for channel in 0..<3 { samples[p+channel] = value * gains[channel] }
        }
        if let (x,y) = transparentPixel {
            let p = (y*192+x)*4
            for channel in 0..<4 { samples[p+channel] = 0 }
        }
        return CIImage(bitmapData: samples.withUnsafeBytes { Data($0) }, bytesPerRow: 192*16,
            size: CGSize(width: 192,height: 192), format: .RGBAf, colorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB))
    }
}

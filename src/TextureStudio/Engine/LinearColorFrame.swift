import CoreImage
import Foundation

/// A stable, full-precision working photograph. Never used to reinterpret source
/// depth, normal or roughness files. Both views share the same linear float data.
struct LinearColorFrame {
    let color: CIImage
    let numeric: CIImage

    init(_ image: CIImage, context: CIContext) throws {
        let extent = image.extent
        guard extent.width.isFinite, extent.height.isFinite,
              extent.width >= 1, extent.height >= 1,
              extent.width * extent.height <= 150_000_000 else {
            throw TextureError.processing("Cannot render this photograph's working colour frame.")
        }
        let width = Int(extent.width), height = Int(extent.height)
        let space = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!
        var pixels = Data(count: width * height * 16)
        pixels.withUnsafeMutableBytes { buffer in
            context.render(image, toBitmap: buffer.baseAddress!, rowBytes: width * 16,
                bounds: extent, format: .RGBAf, colorSpace: space)
        }
        let offset = CGAffineTransform(translationX: extent.minX, y: extent.minY)
        color = CIImage(bitmapData: pixels, bytesPerRow: width * 16,
            size: CGSize(width: width, height: height), format: .RGBAf, colorSpace: space).transformed(by: offset)
        numeric = CIImage(bitmapData: pixels, bytesPerRow: width * 16,
            size: CGSize(width: width, height: height), format: .RGBAf, colorSpace: nil).transformed(by: offset)
    }

    /// Anchor HDR midtones to the corresponding framed SDR photograph. A robust
    /// median ignores clipped highlights and dark divisions, retaining gain-map
    /// detail without turning display headroom into a material exposure change.
    static func exposureGain(reference: CIImage, hdr: CIImage, context: CIContext) -> Float {
        func samples(_ image: CIImage) -> [Float] {
            let side = 128
            let extent = image.extent
            let small = image.transformed(by: CGAffineTransform(translationX: -extent.minX, y: -extent.minY))
                .clampedToExtent().transformed(by: CGAffineTransform(scaleX: CGFloat(side) / extent.width,
                    y: CGFloat(side) / extent.height)).cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
            var values = [Float](repeating: 0, count: side * side * 4)
            values.withUnsafeMutableBytes {
                context.render(small, toBitmap: $0.baseAddress!, rowBytes: side * 16,
                    bounds: small.extent, format: .RGBAf,
                    colorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!)
            }
            return values
        }
        let base = samples(reference), expanded = samples(hdr)
        var ratios: [Float] = []
        var fallback: [Float] = []
        ratios.reserveCapacity(base.count / 4)
        for index in stride(from: 0, to: base.count, by: 4) {
            let baseAlpha = base[index + 3], hdrAlpha = expanded[index + 3]
            guard baseAlpha.isFinite, hdrAlpha.isFinite, baseAlpha > 1e-6, hdrAlpha > 1e-6 else { continue }
            let a = (base[index] * 0.2126 + base[index + 1] * 0.7152 + base[index + 2] * 0.0722) / baseAlpha
            let b = (expanded[index] * 0.2126 + expanded[index + 1] * 0.7152 + expanded[index + 2] * 0.0722) / hdrAlpha
            if a.isFinite, b.isFinite, a > 0.0001, b > 0.00001 {
                let ratio = a / b
                fallback.append(ratio)
                if a > 0.01, a < 0.8 { ratios.append(ratio) }
            }
        }
        if ratios.isEmpty { ratios = fallback }
        guard !ratios.isEmpty else { return 1 }
        ratios.sort()
        return max(1 / 1024, min(1024, ratios[ratios.count / 2]))
    }
}

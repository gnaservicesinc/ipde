import Foundation
import CoreImage
import Vision
import simd

struct PhotoEvidenceResult: @unchecked Sendable {
    let image: CIImage
    let warnings: [String]
}

/// Spatial companions are supporting samples of a surface, never replacement camera calibration.
/// Registration and evidence checks run on bounded grids; the accepted warp samples native images.
enum PhotoEvidenceService {
    static func fuse(source: TextureSource) async throws -> PhotoEvidenceResult {
        guard !source.supportingViews.isEmpty else {
            return PhotoEvidenceResult(image: source.orientedImage, warnings: [])
        }
        try Task.checkCancellation()
        let context = CIContext(options: [.workingFormat: CIFormat.RGBAf,
            .workingColorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!, .cacheIntermediates: false])
        let extent = source.orientedImage.extent
        let scale = min(1, 768 / max(extent.width, extent.height))
        let registrationSize = CGSize(width: max(16, floor(extent.width * scale)),
                                      height: max(16, floor(extent.height * scale)))
        let reference = resized(source.orientedImage, to: registrationSize)
        let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
        guard let referenceCG = context.createCGImage(reference, from: reference.extent, format: .RGBA8, colorSpace: srgb) else {
            return PhotoEvidenceResult(image: source.orientedImage, warnings: ["Could not inspect spatial companion views; kept the primary photo."])
        }
        var image = source.orientedImage
        var notes: [String] = []
        for (index, companion) in source.supportingViews.prefix(2).enumerated() {
            try Task.checkCancellation()
            let floating = resized(companion, to: registrationSize)
            guard let floatingCG = context.createCGImage(floating, from: floating.extent, format: .RGBA8, colorSpace: srgb) else { continue }
            do {
                let request = VNHomographicImageRegistrationRequest(targetedCGImage: floatingCG)
                let handler = VNImageRequestHandler(cgImage: referenceCG)
                try handler.perform([request])
                guard let observation = request.results?.first else {
                    notes.append("Spatial view \(index + 1) had no reliable registration; primary photo retained.")
                    continue
                }
                let matrix = observation.warpTransform
                // Vision works on raster coordinates. Both orientations are checked by actual
                // patch agreement, avoiding any unverified top-left/bottom-left convention.
                let flip = simd_float3x3(columns: (SIMD3(1,0,0), SIMD3(0,-1,0), SIMD3(0,Float(registrationSize.height),1)))
                let candidates = [matrix, flip * matrix * flip]
                var best: (CIImage, EvidenceAssessment)?
                for candidate in candidates {
                    guard let warped = warp(companion, matrix: candidate, registrationSize: registrationSize, extent: extent) else { continue }
                    let assessment = try assess(reference: source.orientedImage, companion: warped, context: context)
                    if best == nil || assessment.acceptedFraction > best!.1.acceptedFraction {
                        best = (warped, assessment)
                    }
                }
                guard let (warped, assessment) = best, assessment.acceptedFraction >= 0.05 else {
                    notes.append("Spatial view \(index + 1) did not match the surface reliably after exposure matching; primary photo retained.")
                    continue
                }
                let corrected = warped.applyingFilter("CIColorMatrix", parameters: [
                    "inputRVector": CIVector(x: CGFloat(assessment.gains[0]), y: 0, z: 0, w: 0),
                    "inputGVector": CIVector(x: 0, y: CGFloat(assessment.gains[1]), z: 0, w: 0),
                    "inputBVector": CIVector(x: 0, y: 0, z: CGFloat(assessment.gains[2]), w: 0),
                    "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                    "inputBiasVector": CIVector(x: CGFloat(assessment.offsets[0]), y: CGFloat(assessment.offsets[1]), z: CGFloat(assessment.offsets[2]), w: 0)
                ])
                image = blendRegisteredCompanion(reference: image, companion: corrected, assessmentMask: assessment.mask)
                notes.append("Spatial view \(index + 1): exposure and color matched; supporting samples blended in \(Int(assessment.acceptedFraction * 100))% of the surface. Uncertain or occluded regions keep the primary photo.")
            } catch is CancellationError { throw CancellationError() }
            catch {
                notes.append("Spatial view \(index + 1) could not be aligned; primary photo retained.")
            }
        }
        return PhotoEvidenceResult(image: image, warnings: notes)
    }

    /// A coarse assessment may miss native-resolution holes in either image.
    /// Color evidence only contributes where both native samples are fully
    /// opaque, so blending cannot fill holes or alter the primary photo's alpha.
    static func blendRegisteredCompanion(reference: CIImage, companion: CIImage, assessmentMask: CIImage) -> CIImage {
        func opaqueCoverage(_ image: CIImage) -> CIImage {
            let coverage = image.applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputGVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputBVector": CIVector(x: 0, y: 0, z: 0, w: 1),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                "inputBiasVector": CIVector(x: 0, y: 0, z: 0, w: 1)
            ])
            // CIColorThreshold uses a strict comparison. The preceding Float32
            // value makes only exact full opacity pass without excluding 1.
            return coverage.applyingFilter("CIColorThreshold", parameters: ["inputThreshold": Float(1).nextDown])
        }
        let extent = reference.extent
        let mask = resized(assessmentMask, to: extent.size)
            .transformed(by: CGAffineTransform(translationX: extent.minX, y: extent.minY))
            .applyingFilter("CIMultiplyCompositing", parameters: [kCIInputBackgroundImageKey: opaqueCoverage(reference)])
            .applyingFilter("CIMultiplyCompositing", parameters: [kCIInputBackgroundImageKey: opaqueCoverage(companion)])
        return companion.applyingFilter("CIBlendWithMask", parameters: [kCIInputBackgroundImageKey: reference, kCIInputMaskImageKey: mask])
            .cropped(to: extent)
    }

    private static func resized(_ image: CIImage, to size: CGSize) -> CIImage {
        let zero = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        return zero.transformed(by: CGAffineTransform(scaleX: size.width / zero.extent.width, y: size.height / zero.extent.height))
    }

    private static func warp(_ image: CIImage, matrix: simd_float3x3, registrationSize: CGSize, extent: CGRect) -> CIImage? {
        func project(_ x: CGFloat, _ y: CGFloat) -> CGPoint? {
            let p = matrix * SIMD3(Float(x), Float(y), 1)
            guard p.z.isFinite, abs(p.z) > 1e-6 else { return nil }
            let point = CGPoint(x: CGFloat(p.x / p.z) * extent.width / registrationSize.width,
                                y: CGFloat(p.y / p.z) * extent.height / registrationSize.height)
            guard point.x.isFinite, point.y.isFinite, abs(point.x) < extent.width * 8, abs(point.y) < extent.height * 8 else { return nil }
            return point
        }
        guard let bl = project(0,0), let br = project(registrationSize.width,0),
              let tl = project(0,registrationSize.height), let tr = project(registrationSize.width,registrationSize.height) else { return nil }
        let warped = image.applyingFilter("CIPerspectiveTransform", parameters: ["inputBottomLeft": CIVector(cgPoint: bl), "inputBottomRight": CIVector(cgPoint: br), "inputTopLeft": CIVector(cgPoint: tl), "inputTopRight": CIVector(cgPoint: tr)])
        // cropped(to:) intersects extents; a translated view can therefore return a
        // smaller frame. Assessment must keep the reference coordinates, rather than
        // stretching that intersection and silently undoing the registration.
        let canvas = CIImage(color:CIColor.clear).cropped(to:extent)
        return warped.composited(over:canvas).cropped(to:extent)
    }

    struct EvidenceAssessment {
        let gains: [Float]
        let offsets: [Float]
        let mask: CIImage
        let acceptedFraction: Double
    }

    /// Fit exposure/color using unclipped samples, then require textured patch agreement.
    /// The mask has a one-cell zero border around every accepted region, preventing blur
    /// interpolation from introducing companion content into rejected regions.
    static func assess(reference: CIImage, companion: CIImage, context: CIContext) throws -> EvidenceAssessment {
        let side = 192
        let size = CGSize(width: side, height: side)
        let linear = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!
        func pixels(_ image: CIImage) -> [Float] {
            var values = [Float](repeating: 0, count: side * side * 4)
            let small = resized(image, to: size)
            values.withUnsafeMutableBytes {
                context.render(small, toBitmap: $0.baseAddress!, rowBytes: side * 16, bounds: CGRect(origin: .zero, size: size), format: .RGBAf, colorSpace: linear)
            }
            return values
        }
        let a = pixels(reference), b = pixels(companion)
        var gains = [Float](repeating: 1, count: 3)
        var offsets = [Float](repeating: 0, count: 3)
        for channel in 0..<3 {
            var n: Float = 0, sx: Float = 0, sy: Float = 0, sxx: Float = 0, sxy: Float = 0
            for p in stride(from: 0, to: a.count, by: 4) {
                let x = b[p + channel], y = a[p + channel]
                guard b[p+3] > 0.99, x.isFinite, y.isFinite, x > 0.005, x < 0.95, y > 0.005, y < 0.95 else { continue }
                n += 1; sx += x; sy += y; sxx += x*x; sxy += x*y
            }
            guard n > 64 else { continue }
            let variance = sxx - sx*sx/n
            guard variance > 1e-5 else { continue }
            gains[channel] = min(4, max(0.25, (sxy-sx*sy/n)/variance))
            offsets[channel] = min(0.15, max(-0.15, (sy-gains[channel]*sx)/n))
        }
        var mask = [Float](repeating: 0, count: side * side)
        let patch = 12
        var accepted = 0
        for py in stride(from: 0, to: side, by: patch) {
            try Task.checkCancellation()
            for px in stride(from: 0, to: side, by: patch) {
                var n: Float = 0, sa: Float = 0, sb: Float = 0, saa: Float = 0, sbb: Float = 0, sab: Float = 0, error: Float = 0
                for y in py..<min(side, py+patch) {
                    for x in px..<min(side, px+patch) {
                        let p = (y*side+x)*4
                        guard b[p+3] > 0.99 else { continue }
                        let av = a[p]*0.2126 + a[p+1]*0.7152 + a[p+2]*0.0722
                        let bv = (b[p]*gains[0]+offsets[0])*0.2126 + (b[p+1]*gains[1]+offsets[1])*0.7152 + (b[p+2]*gains[2]+offsets[2])*0.0722
                        guard av.isFinite, bv.isFinite else { continue }
                        n += 1; sa += av; sb += bv; saa += av*av; sbb += bv*bv; sab += av*bv
                        error += abs(av-bv)
                    }
                }
                guard n >= Float(patch*patch)*0.95 else { continue }
                let va = saa-sa*sa/n, vb = sbb-sb*sb/n
                guard va > n*0.0001, vb > n*0.0001 else { continue }
                let correlation = (sab-sa*sb/n)/sqrt(va*vb)
                guard correlation >= 0.9, error/n < 0.035 else { continue }
                // Keep a strong primary anchor, regardless of secondary exposure or resolution.
                for y in (py+1)..<min(side-1,py+patch-1) {
                    for x in (px+1)..<min(side-1,px+patch-1) {
                        let p = (y*side+x)*4
                        guard a[p+3] > 0.99, b[p+3] > 0.99,
                              (0..<3).allSatisfy({ a[p+$0].isFinite && b[p+$0].isFinite }) else { continue }
                        mask[y*side+x] = 0.3
                        accepted += 1
                    }
                }
            }
        }
        // CIBlendWithMask samples RGB luminance. A single-channel .Rf source may be
        // sampled as (R,0,0), even though render(...format:.Rf) looks correct. Expand
        // the scalar mask to all three channels so actual native fusion gets its weight.
        var rgbaMask = [Float](repeating:1,count:side*side*4)
        for index in mask.indices {
            for channel in 0..<3 { rgbaMask[index*4+channel] = mask[index] }
        }
        let maskImage = CIImage(bitmapData: rgbaMask.withUnsafeBytes { Data($0) }, bytesPerRow: side*16,
                                size: size, format: .RGBAf, colorSpace: nil)
        return EvidenceAssessment(gains: gains, offsets: offsets, mask: maskImage, acceptedFraction: Double(accepted)/Double(side*side))
    }
}

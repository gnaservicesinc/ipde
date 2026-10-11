import XCTest
import CoreImage
import ImageIO
import Metal
@testable import TextureStudio

/// Colour rendering regressions use the complete material pipeline, including its
/// 2K blur and Metal kernels. The optional RAW fixture stays outside the repository.
final class DiffuseRenderingTests: XCTestCase {
    private let linear = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!
    private let srgb = CGColorSpace(name: CGColorSpace.sRGB)!

    func testUniformHDRGainPreservesBasePhotoExposureAt2048() async throws {
        let side = 2048
        let base: [Float] = [0.28, 0.18, 0.10, 1]
        let boosted: [Float] = [base[0] * 8, base[1] * 8, base[2] * 8, 1]
        let source = TextureSource(url: URL(fileURLWithPath: "/tmp/uniform-hdr-surface.png"),
            orientedImage: constant(base, side: side),
            camera: CameraMetadata(), pixelWidth: side, pixelHeight: side,
            hdrImage: constant(boosted, side: side))
        let engine = TextureEngine()
        var settings = TextureSettings()
        settings.outputSize = side
        let hdr = try await engine.process(source: source, settings: settings)
        settings.useHDRGainMap = false
        let sdr = try await engine.process(source: source, settings: settings)
        let context = try colourContext()

        // A uniform gain map carries no extra surface contrast. It must therefore
        // leave an otherwise identical photograph at its original SDR exposure.
        // Sampling across tiles also catches local failures hidden by one mean.
        for (x, y) in [(64, 64), (512, 1024), (1024, 1024), (1536, 768), (1984, 1984)] {
            let actual = pixel(hdr.diffuse, x: x, y: y, context: context)
            let expected = pixel(sdr.diffuse, x: x, y: y, context: context)
            for channel in 0..<3 {
                XCTAssertTrue(actual[channel].isFinite)
                XCTAssertEqual(actual[channel], expected[channel], accuracy: 0.005,
                    "Uniform HDR gain changed channel \(channel) exposure at \(x),\(y)")
            }
            XCTAssertEqual(actual[3], 1, accuracy: 0.0001)
        }
    }

    func testPerspectiveCropPreservesInteriorTransparencyWithoutChangingVisibleColorOrGeometry() async throws {
        let side = 256, color: [Float] = [0.28, 0.18, 0.10]
        var rgba = [Float](repeating: 0, count: side * side * 4)
        for y in 0..<side { for x in 0..<side {
            let alpha: Float = hypot(Double(x - side / 2), Double(y - side / 2)) < 40 ? 0 : 0.35
            let offset = (y * side + x) * 4
            for channel in 0..<3 { rgba[offset + channel] = color[channel] * alpha }
            rgba[offset + 3] = alpha
        } }
        let transparent = CIImage(bitmapData: rgba.withUnsafeBytes { Data($0) }, bytesPerRow: side * 16,
                                  size: CGSize(width: side, height: side), format: .RGBAf, colorSpace: linear)
        func source(_ image: CIImage) -> TextureSource {
            TextureSource(url: URL(fileURLWithPath: "/tmp/transparent-surface.png"), orientedImage: image,
                          camera: CameraMetadata(), pixelWidth: side, pixelHeight: side)
        }
        let engine = TextureEngine(), context = try colourContext()
        var settings = TextureSettings()
        settings.rotationX = 20; settings.rotationY = 10; settings.rotationZ = 4
        settings.heightDetail = 0.75; settings.roughnessDetail = 1
        let opaque = try await engine.prepareDiffuse(source: source(constant(color + [1], side: side)), settings: settings)
        let prepared = try await engine.prepareDiffuse(source: source(transparent), settings: settings)
        XCTAssertEqual(prepared.crop, opaque.crop, "An interior transparency hole must not shrink or move the geometric crop")
        XCTAssertEqual(prepared.corners, opaque.corners)
        let material = try await engine.process(source: source(transparent), settings: settings, preparedDiffuse: prepared)
        let centre = pixel(material.diffuse, x: 512, y: 512, context: context)
        XCTAssertEqual(centre[3], 0, accuracy: 0.0001, "A valid projected source pixel may still be transparent")
        for channel in 0..<3 { XCTAssertEqual(centre[channel], 0, accuracy: 0.0001) }
        for (x, y) in [(100, 100), (900, 100), (100, 900), (900, 900)] {
            let actual = pixel(material.diffuse, x: x, y: y, context: context)
            let reference = pixel(opaque.diffuse, x: x, y: y, context: context)
            XCTAssertEqual(actual[3], 0.35, accuracy: 0.0001)
            for channel in 0..<3 {
                XCTAssertEqual(actual[channel] / actual[3], reference[channel], accuracy: 0.001,
                               "Opacity must not change a visible material's straight color")
            }
            XCTAssertEqual(pixel(material.height, x: x, y: y, context: context)[0], 0.5, accuracy: 0.001)
            XCTAssertEqual(pixel(material.roughness, x: x, y: y, context: context)[0], settings.roughnessBase, accuracy: 0.001)
        }
        XCTAssertEqual(pixel(material.height, x: 512, y: 512, context: context)[0], 0.5, accuracy: 0.001,
                       "Fully transparent photo pixels must not fabricate brightness relief")
        XCTAssertEqual(pixel(material.roughness, x: 512, y: 512, context: context)[0], settings.roughnessBase, accuracy: 0.001)
    }

    func testOptInBrownWallRAWHasFiniteDiffuseWithoutSaturatedRedTiles() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["TEXTURE_STUDIO_TEST_PHOTO"], !path.isEmpty else {
            throw XCTSkip("Set TEXTURE_STUDIO_TEST_PHOTO to the original IMG_1951.DNG brown-wall fixture.")
        }
        guard path.hasPrefix("/") else {
            XCTFail("TEXTURE_STUDIO_TEST_PHOTO must be an absolute path")
            return
        }
        let photoURL = URL(fileURLWithPath: path)
        XCTAssertTrue(FileManager.default.fileExists(atPath: photoURL.path))
        let output: URL
        if let outputPath = environment["TEXTURE_STUDIO_TEST_OUTPUT"], !outputPath.isEmpty {
            guard outputPath.hasPrefix("/") else {
                XCTFail("TEXTURE_STUDIO_TEST_OUTPUT must be an absolute path")
                return
            }
            output = URL(fileURLWithPath: outputPath, isDirectory: true)
        } else {
            output = FileManager.default.temporaryDirectory
                .appendingPathComponent("diffuse-diagnostic-\(UUID().uuidString)", isDirectory: true)
        }
        // Leave these derived images available for visual inspection. Never write
        // into the photo directory or alter the original RAW file.
        XCTAssertNotEqual(output.standardizedFileURL, photoURL.deletingLastPathComponent().standardizedFileURL)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        let engine = TextureEngine()
        let source = try await engine.importPhoto(photoURL)
        XCTAssertNotNil(source.hdrImage, "The regression fixture must exercise Apple's RAW/HDR decoding path")
        var settings = TextureSettings()
        settings.outputSize = 2048
        let context = try colourContext()
        var reports: [[String: Any]] = []
        for (name, enabled) in [("diffuse-hdr-default-2048", true), ("diffuse-sdr-2048", false)] {
            settings.useHDRGainMap = enabled
            let material = try await engine.process(source: source, settings: settings)
            XCTAssertEqual(material.diffuse.extent.size, CGSize(width: 2048, height: 2048))
            // Inspect the first exported PNG, before preview/float rendering can
            // warm the RAW graph and hide the first-write corruption.
            let folder = output.appendingPathComponent(name, isDirectory: true)
            _ = try await engine.export(material, to: folder, precision: .float32)
            let pngURL = folder.appendingPathComponent("diffuse.png")
            let pngSource = try XCTUnwrap(CGImageSourceCreateWithURL(pngURL as CFURL, nil))
            let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(pngSource, 0, nil))
            let report = try metrics(material.diffuse, preview: image, context: context)
            XCTAssertEqual(report.invalidSamples, 0, "\(name) contains invalid colour samples")
            XCTAssertLessThanOrEqual(report.saturatedRedPixels, 64,
                "\(name) introduced red pixels into the neutral brown-wall fixture")
            XCTAssertLessThan(report.maximumRedTileFraction, 0.02,
                "\(name) contains a saturated-red tile, like the reported rectangular artefact")
            let values: [String: Any] = ["image": pngURL.path,
                "hdrEnabled": enabled, "minimumRGB": report.minimum,
                "maximumRGB": report.maximum, "meanRGB": report.mean,
                "invalidSamples": report.invalidSamples,
                "clippedPixels": report.clippedPixels,
                "saturatedRedPixels": report.saturatedRedPixels,
                "maximum128PixelRedTileFraction": report.maximumRedTileFraction]
            reports.append(values)
            print("DIFFUSE_DIAGNOSTIC \(values)")
        }
        let reportURL = output.appendingPathComponent("diffuse-rendering-report.json")
        let data = try JSONSerialization.data(withJSONObject: ["source": photoURL.path,
            "sourcePixels": [source.pixelWidth, source.pixelHeight],
            "auxiliaryTypes": source.camera.auxiliaryTypes, "results": reports],
            options: [.prettyPrinted, .sortedKeys])
        try data.write(to: reportURL, options: .atomic)
        print("DIFFUSE_DIAGNOSTIC_REPORT=\(reportURL.path)")
    }

    private func constant(_ rgba: [Float], side: Int) -> CIImage {
        CIImage(bitmapData: rgba.withUnsafeBytes { Data($0) }, bytesPerRow: 16,
            size: CGSize(width: 1, height: 1), format: .RGBAf, colorSpace: linear)
            .clampedToExtent().cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
    }

    private func colourContext() throws -> CIContext {
        guard let metal = MTLCreateSystemDefaultDevice() else { throw XCTSkip("Metal is unavailable") }
        return CIContext(mtlDevice: metal, options: [.workingFormat: CIFormat.RGBAf,
            .workingColorSpace: linear, .cacheIntermediates: false])
    }

    private func pixel(_ image: CIImage, x: Int, y: Int, context: CIContext) -> [Float] {
        var rgba = [Float](repeating: 0, count: 4)
        rgba.withUnsafeMutableBytes {
            context.render(image, toBitmap: $0.baseAddress!, rowBytes: 16,
                bounds: CGRect(x: x, y: y, width: 1, height: 1), format: .RGBAf, colorSpace: linear)
        }
        return rgba
    }

    private struct RenderMetrics {
        let minimum: [Float]
        let maximum: [Float]
        let mean: [Double]
        let invalidSamples: Int
        let clippedPixels: Int
        let saturatedRedPixels: Int
        let maximumRedTileFraction: Double
    }

    private func metrics(_ image: CIImage, preview: CGImage, context: CIContext) throws -> RenderMetrics {
        let width = preview.width, height = preview.height
        let pixelCount = width * height
        var values = [Float](repeating: 0, count: pixelCount * 4)
        values.withUnsafeMutableBytes {
            context.render(image, toBitmap: $0.baseAddress!, rowBytes: width * 16,
                bounds: image.extent, format: .RGBAf, colorSpace: linear)
        }
        var minimum = [Float](repeating: .greatestFiniteMagnitude, count: 3)
        var maximum = [Float](repeating: -.greatestFiniteMagnitude, count: 3)
        var totals = [Double](repeating: 0, count: 3)
        var invalid = 0, clipped = 0
        for p in 0..<pixelCount {
            var pixelClipped = false
            for c in 0..<4 {
                let sample = values[p * 4 + c]
                guard sample.isFinite else { invalid += 1; continue }
                if c < 3 {
                    minimum[c] = min(minimum[c], sample)
                    maximum[c] = max(maximum[c], sample)
                    totals[c] += Double(sample)
                    if sample >= 0.9999 { pixelClipped = true }
                }
            }
            if pixelClipped { clipped += 1 }
        }
        var bytes = [UInt8](repeating: 0, count: pixelCount * 4)
        try bytes.withUnsafeMutableBytes { buffer in
            let bitmap = try XCTUnwrap(CGContext(data: buffer.baseAddress!, width: width,
                height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: srgb,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue))
            bitmap.draw(preview, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        let tileSide = 128, columns = (width + tileSide - 1) / tileSide
        let rows = (height + tileSide - 1) / tileSide
        var tileReds = [Int](repeating: 0, count: columns * rows)
        var saturatedRed = 0
        for y in 0..<height {
            for x in 0..<width {
                let p = (y * width + x) * 4
                if bytes[p] > 220 && bytes[p + 1] < 32 && bytes[p + 2] < 96 {
                    saturatedRed += 1
                    tileReds[(y / tileSide) * columns + x / tileSide] += 1
                }
            }
        }
        let maximumFraction = tileReds.enumerated().map { index, count in
            let tileWidth = min(tileSide, width - (index % columns) * tileSide)
            let tileHeight = min(tileSide, height - (index / columns) * tileSide)
            return Double(count) / Double(tileWidth * tileHeight)
        }.max() ?? 0
        return RenderMetrics(minimum: minimum, maximum: maximum,
            mean: totals.map { $0 / Double(pixelCount) }, invalidSamples: invalid,
            clippedPixels: clipped, saturatedRedPixels: saturatedRed,
            maximumRedTileFraction: maximumFraction)
    }
}

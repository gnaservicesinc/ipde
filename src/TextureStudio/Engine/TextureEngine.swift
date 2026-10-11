import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
import ImageIO
import Metal
import Darwin

actor TextureEngine {
    private let context: CIContext
    private let dataContext: CIContext
    private let device: MTLDevice?
    private let libraryURL: URL?
    private let linear = CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!
    private let srgb = CGColorSpace(name: CGColorSpace.sRGB)!
    private var kernels: Kernels?
    private var preparedHDR: (source: CIImage, color: CIImage)?

    init(libraryURL: URL? = nil) {
        let metal = MTLCreateSystemDefaultDevice()
        self.device = metal
        self.libraryURL = libraryURL
        let options: [CIContextOption: Any] = [.workingFormat: CIFormat.RGBAf,
            .workingColorSpace: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!,
            .cacheIntermediates: false, .name: "Texture Studio colour"]
        let dataOptions: [CIContextOption: Any] = [.workingFormat: CIFormat.RGBAf,
            .workingColorSpace: NSNull(), .outputColorSpace: NSNull(),
            .cacheIntermediates: false, .name: "Texture Studio float data"]
        self.context = metal.map { CIContext(mtlDevice: $0, options: options) } ?? CIContext(options: options)
        self.dataContext = metal.map { CIContext(mtlDevice: $0, options: dataOptions) } ?? CIContext(options: dataOptions)
    }

    /// Photo auxiliaries can help decode colour and identify masks. Apple's
    /// portrait depth/disparity describes object placement, not material relief,
    /// so Studio neither decodes it nor advertises it as an available map.
    static var materialPhotoAuxiliaryTypes: [(CFString, String)] {
        [(kCGImageAuxiliaryDataTypeHDRGainMap, "HDR gain map"),
         (kCGImageAuxiliaryDataTypeISOGainMap, "ISO HDR gain map"),
         (kCGImageAuxiliaryDataTypePortraitEffectsMatte, "Portrait matte"),
         (kCGImageAuxiliaryDataTypeSemanticSegmentationHairMatte, "Hair matte"),
         (kCGImageAuxiliaryDataTypeSemanticSegmentationSkinMatte, "Skin matte"),
         (kCGImageAuxiliaryDataTypeSemanticSegmentationSkyMatte, "Sky matte")]
    }

    func importPhoto(_ url: URL) throws -> TextureSource {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache:false] as CFDictionary) else {
            throw TextureError.invalidImage("Cannot open this photo. Choose a supported HEIC, JPEG, PNG, TIFF, or RAW image.")
        }
        let primaryIndex = CGImageSourceGetPrimaryImageIndex(source)
        guard let properties = CGImageSourceCopyPropertiesAtIndex(source, primaryIndex, nil) as? [String: Any] else {
            throw TextureError.invalidImage("Cannot read this photo's primary image.")
        }
        let image = CIImage(cgImageSource:source,index:primaryIndex,options:[.applyOrientationProperty:true])
        let width = image.extent.width, height = image.extent.height
        guard width.isFinite, height.isFinite, width >= 4, height >= 4,
              width * height <= 150_000_000 else {
            throw TextureError.invalidImage("The photo is empty or exceeds the 150-megapixel import limit.")
        }
        let exif = properties[kCGImagePropertyExifDictionary as String] as? [String: Any] ?? [:]
        let tiff = properties[kCGImagePropertyTIFFDictionary as String] as? [String: Any] ?? [:]
        var metadata = CameraMetadata()
        metadata.make = tiff[kCGImagePropertyTIFFMake as String] as? String
        metadata.model = tiff[kCGImagePropertyTIFFModel as String] as? String
        metadata.lensModel = exif[kCGImagePropertyExifLensModel as String] as? String
        metadata.focalLengthMillimeters = (exif[kCGImagePropertyExifFocalLength as String] as? NSNumber)?.doubleValue
        metadata.focalLength35mm = (exif[kCGImagePropertyExifFocalLenIn35mmFilm as String] as? NSNumber)?.doubleValue
        metadata.aperture = (exif[kCGImagePropertyExifFNumber as String] as? NSNumber)?.doubleValue
        metadata.exposureSeconds = (exif[kCGImagePropertyExifExposureTime as String] as? NSNumber)?.doubleValue
        metadata.iso = (exif[kCGImagePropertyExifISOSpeedRatings as String] as? [NSNumber])?.first?.doubleValue
        metadata.orientation = (properties[kCGImagePropertyOrientation as String] as? NSNumber)?.uint32Value ?? 1
        metadata.sourceBitDepth = (properties[kCGImagePropertyDepth as String] as? NSNumber)?.intValue
        metadata.colorProfile = properties[kCGImagePropertyProfileName as String] as? String
        for (type, label) in Self.materialPhotoAuxiliaryTypes {
            guard CGImageSourceCopyAuxiliaryDataInfoAtIndex(source, primaryIndex, type) != nil else { continue }
            metadata.auxiliaryTypes.append(label)
        }
        let zero = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        var hdrImage: CIImage?
        if metadata.auxiliaryTypes.contains("HDR gain map") || metadata.auxiliaryTypes.contains("ISO HDR gain map"),
           let expanded = CIImage(contentsOf:url,options:[.applyOrientationProperty:true,.expandToHDR:true]),
           expanded.extent.size == zero.extent.size {
            hdrImage = expanded.transformed(by:CGAffineTransform(translationX:-expanded.extent.minX,y:-expanded.extent.minY))
        }
        let indices = supportingIndices(source,primaryIndex:primaryIndex)
        var acceptedIndices = [Int]()
        let supporting = indices.compactMap { index -> CIImage? in
            let view = CIImage(cgImageSource:source,index:index,options:[.applyOrientationProperty:true])
            guard view.extent.width >= 32,view.extent.height >= 32,
                  view.extent.width*view.extent.height <= 150_000_000 else { return nil }
            acceptedIndices.append(index)
            return view.transformed(by:CGAffineTransform(translationX:-view.extent.minX,y:-view.extent.minY))
        }
        return TextureSource(url: url, orientedImage: zero,
                             camera: metadata, pixelWidth: Int(width), pixelHeight: Int(height),
                             supportingViews:supporting,primaryImageIndex:primaryIndex,supportingImageIndices:acceptedIndices,
                             hdrImage:hdrImage)
    }

    func importDepth(_ url: URL, matching source: TextureSource) throws -> TextureDepth {
        guard let image = CIImage(contentsOf: url, options: [.colorSpace:NSNull(), .applyOrientationProperty:true]),
              image.extent.width > 0, image.extent.height > 0,
              image.extent.width * image.extent.height <= 150_000_000 else {
            throw TextureError.invalidDepth("Cannot read this depth map. Choose a float EXR or TIFF registered to the photo.")
        }
        let ratio = image.extent.width / image.extent.height
        let sourceRatio = Double(source.pixelWidth) / Double(source.pixelHeight)
        guard abs(ratio/sourceRatio - 1) < 0.02 else {
            throw TextureError.invalidDepth("Depth and photo aspect ratios differ. Attach a map registered to this photo's full frame.")
        }
        return TextureDepth(image: image, sourceLabel: "Attached depth: \(url.lastPathComponent)")
    }

    struct PreparedDiffuse: @unchecked Sendable {
        let diffuse: CIImage
        let numeric: CIImage
        let extent: CGRect
        let output: CGRect
        let corners: [CGPoint]
        let crop: CGRect
        let usesHDR: Bool
        let warnings: [String]
    }

    func prepareDiffuse(source: TextureSource, settings: TextureSettings) async throws -> PreparedDiffuse {
        guard device != nil else { throw TextureError.missingMetal }
        try Task.checkCancellation()
        try validate(settings)
        let kernels = try loadKernels()
        let extent = source.orientedImage.extent
        let focal = settings.focalLengthPixels ?? source.camera.focalLength35mm.map {
            max(extent.width,extent.height) * $0 / 36
        } ?? max(extent.width,extent.height) * 1.35
        let corners = try TextureGeometry.projectedCorners(width: extent.width, height: extent.height,
            xDegrees: settings.rotationX, yDegrees: settings.rotationY, zDegrees: settings.rotationZ, focalPixels: focal)
        // Source alpha describes material opacity, including interior holes.
        // Only the projected full-frame geometry determines crop coverage.
        let largest = try TextureGeometry.maximumCrop(in: corners)
        let crop = try TextureGeometry.framedCrop(largest, scale: settings.cropScale,
                                                 offsetX: settings.cropOffsetX, offsetY: settings.cropOffsetY)
        let output = CGRect(x:0,y:0,width:settings.outputSize,height:settings.outputSize)
        let usesHDR = settings.useHDRGainMap && source.hdrImage != nil
        let processingPhoto: CIImage
        if usesHDR, let hdr = source.hdrImage {
            if preparedHDR?.source === hdr { processingPhoto = preparedHDR!.color }
            else {
                // Apple's lazy RAW/HDR decoding can change exposure under ROI
                // cropping. Decode the complete frame once before any transform.
                let color = try LinearColorFrame(hdr, context: context).color
                preparedHDR = (hdr, color)
                processingPhoto = color
            }
        } else {
            preparedHDR = nil
            processingPhoto = source.orientedImage
        }
        let evidence: PhotoEvidenceResult
        if settings.useSupportingViews, !source.supportingViews.isEmpty {
            var evidenceSource = source
            if usesHDR {
                evidenceSource = TextureSource(url:source.url,orientedImage:processingPhoto,
                    camera:source.camera,pixelWidth:source.pixelWidth,pixelHeight:source.pixelHeight,
                    supportingViews:source.supportingViews,primaryImageIndex:source.primaryImageIndex,
                    supportingImageIndices:source.supportingImageIndices,hdrImage:source.hdrImage)
            }
            evidence = try await PhotoEvidenceService.fuse(source:evidenceSource)
        } else { evidence = PhotoEvidenceResult(image:processingPhoto,warnings:[]) }
        let lensPhoto = try lensCorrect(evidence.image, amount:settings.lensDistortion, kernels:kernels)
        var warped = transform(lensPhoto, corners: corners, crop: crop, size: settings.outputSize)
        if usesHDR {
            let reference = transform(try lensCorrect(source.orientedImage, amount: settings.lensDistortion, kernels: kernels),
                corners: corners, crop: crop, size: settings.outputSize)
            let gain = LinearColorFrame.exposureGain(reference: reference, hdr: warped, context: context)
            warped = warped.applyingFilter("CIExposureAdjust", parameters: [kCIInputEVKey: log2(gain)])
        }
        // Keep the photographed high-frequency detail intact. Capture noise can
        // be reduced with registered companion evidence, never a spatial denoiser.
        let photo = warped
        let lighting = photo.clampedToExtent().applyingGaussianBlur(sigma: Double(settings.lightingRadius) * Double(settings.outputSize)).cropped(to:output)
        let target = try meanLuminance(photo)
        guard let diffuseGraph = kernels.delight.apply(extent:output, arguments:[photo,lighting,target,settings.lightingStrength,usesHDR ? Float(1) : Float(0)]) else {
            throw TextureError.processing("Metal lighting correction failed.")
        }
        // First-write PNG evaluation of a lazy HDR graph can introduce a red
        // tile. Freeze the linear result before encoding or numeric-map work.
        let diffuseFrame = try LinearColorFrame(diffuseGraph, context: context)
        return PreparedDiffuse(diffuse: diffuseFrame.color, numeric: diffuseFrame.numeric,
            extent: extent, output: output, corners: corners, crop: crop, usesHDR: usesHDR, warnings: evidence.warnings)
    }

    func writeDiffuse(_ image: CIImage, to url: URL) throws {
        try context.writePNGRepresentation(of: image, to: url, format: .RGBA16, colorSpace: srgb)
    }

    func process(source: TextureSource, settings: TextureSettings,
                 attachedDepth: TextureDepth? = nil, preparedDiffuse: PreparedDiffuse? = nil,
                 modelMaps: [String: MaterialModelMap] = [:]) async throws -> MaterialResult {
        guard device != nil else { throw TextureError.missingMetal }
        try validate(settings)
        let prepared: PreparedDiffuse
        if let preparedDiffuse { prepared = preparedDiffuse }
        else { prepared = try await prepareDiffuse(source: source, settings: settings) }
        let kernels = try loadKernels()
        let extent = prepared.extent, output = prepared.output, corners = prepared.corners, crop = prepared.crop
        for (target, map) in modelMaps {
            guard ["roughness", "normal"].contains(target), map.target == target, map.image.extent == output else {
                throw TextureError.processing("The selected \(target) model must return the exact prepared diffuse grid.")
            }
        }
        let usesHDR = prepared.usesHDR, diffuse = prepared.diffuse, linearDiffuse = prepared.numeric
        let detailRadius = max(2.0, Double(settings.outputSize) * 0.015)
        let lowPhoto = linearDiffuse.clampedToExtent().applyingGaussianBlur(sigma:detailRadius).cropped(to:output)
        var depthOrigin = "Flat relief: no surface depth supplied"
        var warnings = ["Roughness is an editable contrast estimate; a single photograph does not determine physical roughness.",
                        "Broad illumination correction cannot recover clipped highlights or fully remove hard cast shadows.",
                        "Height is relative material relief, with neutral level 0.5; originals and source depth remain unchanged.",
                        "Maps are not automatically seamless. Inspect edges before using a repeating material."]
        warnings.append(contentsOf:prepared.warnings)
        if usesHDR { warnings.append("HDR detail is decoded before cropping, with midtone exposure matched to the base photo and a smooth highlight rolloff for 16-bit sRGB PNG. Source data stays untouched.") }
        var baseHeight = CIImage(color:CIColor(red:0.5,green:0.5,blue:0.5)).cropped(to:output)
        if let selected = attachedDepth {
            let aligned = selected.alignedToOutput ? selected.image : try lensCorrect(resize(selected.image,to:extent.size),amount:settings.lensDistortion,kernels:kernels)
            if selected.interpretation == .surfaceHeight {
                // Supplied surface height already carries its intended relief.
                // Camera-distance normalization would destroy that amplitude.
                let native = selected.alignedToOutput ? resize(aligned, to: output.size) : transform(aligned,corners:corners,crop:crop,size:settings.outputSize)
                let gain = settings.heightStrength * (settings.heightInvert ? -1 : 1)
                let offset = Float(0.5) * (1 - gain)
                baseHeight = native.applyingFilter("CIColorMatrix", parameters:[
                    "inputRVector":CIVector(x:CGFloat(gain),y:0,z:0,w:0),
                    "inputGVector":CIVector(x:CGFloat(gain),y:0,z:0,w:0),
                    "inputBVector":CIVector(x:CGFloat(gain),y:0,z:0,w:0),
                    "inputBiasVector":CIVector(x:CGFloat(offset),y:CGFloat(offset),z:CGFloat(offset),w:0)])
                depthOrigin = selected.sourceLabel
                warnings.append("Supplied surface height keeps its original range and amplitude. Perspective/crop and explicit relief contrast apply; camera-depth normalization, plane removal and depth cleanup are bypassed.")
            } else {
            // Keep existing prediction detail through cleanup at the selected
            // output grid rather than silently reducing a 4K/8K prediction.
            let side=Self.depthCleanupSide(outputSize:settings.outputSize, depthExtent:selected.image.extent)
            let depth=transform(aligned,corners:corners,crop:crop,size:side)
            var values=[Float](repeating:0,count:side*side)
            values.withUnsafeMutableBytes { dataContext.render(depth,toBitmap:$0.baseAddress!,rowBytes:side*4,
                bounds:depth.extent,format:.Rf,colorSpace:nil) }
            let processed=try SurfaceHeightProcessor.derive(values:values,width:side,height:side,
                interpretation:selected.interpretation,planeRemoval:settings.surfacePlaneRemoval,
                cleanup:settings.depthCleanup,strength:settings.heightStrength,invert:settings.heightInvert,
                adaptive:settings.adaptiveRelief)
            let materialGrid=CIImage(bitmapData:processed.values.withUnsafeBytes { Data($0) },bytesPerRow:side*4,
                size:CGSize(width:side,height:side),format:.Rf,colorSpace:nil)
            baseHeight=resize(materialGrid,to:output.size)
            depthOrigin=selected.sourceLabel
            warnings.append("Surface height is derived from registered depth, with global plane removal and depth-only cleanup. It retains broad relief instead of depth-edge halos.")
            warnings.append("Relative height uses a robust depth range; its physical displacement scale is an artistic control, not measured texture depth.")
            if processed.hasRelief && processed.amplitudeGain < 0.999 {
                warnings.append("Near-flat surface protection reduces model relief to \(Int((processed.amplitudeGain*100).rounded()))% of the normalized range to avoid amplifying weak relative depth variation. This is an artistic safeguard, not model confidence; disable it to restore the full range.")
            }
            if !processed.hasRelief { warnings.append("The selected depth contains no relief after plane removal. A neutral height is used; photograph colours do not create bumps.") }
            if processed.repairedSamples > 0 { warnings.append("\(processed.repairedSamples) invalid depth samples were filled from neighbouring depth for the derived material; the source map is untouched.") }
            if settings.outputSize > side { warnings.append("Height is cleaned on a \(side) × \(side) prediction grid, then resampled for export; a larger map does not reveal finer geometry.") }
            }
        } else {
            warnings.append("No material depth is selected. Displacement stays flat unless you explicitly enable artistic brightness relief.")
        }
        if settings.heightDetail > 0 {
            warnings.append("Artistic brightness relief is enabled. Albedo patterns and remaining shadows may create false bumps; this is not model-predicted geometry.")
            if attachedDepth == nil { depthOrigin="Artistic brightness relief (opt-in; colours are not measured height)" }
        }
        if settings.focalLengthPixels == nil && source.camera.focalLength35mm == nil &&
            (abs(settings.rotationX) > 0 || abs(settings.rotationY) > 0) {
            warnings.append("Perspective uses an estimated focal length. Set focal pixels for closer camera matching.")
        }
        if Double(settings.outputSize) > crop.width {
            warnings.append("Output exceeds the crop's \(Int(crop.width)) native pixels per side; upsampling adds no source detail.")
        }
        if settings.outputSize >= 4098 { warnings.append("Export rendering adapts its tile size to available memory. Preview size and the selected ML inference resolution remain independent of the final output size.") }
        if settings.lensDistortion != 0 { warnings.append("Manual radial lens correction uses a safe centre zoom; it is not a calibrated lens profile.") }
        let heightCandidate = attachedDepth?.interpretation == .surfaceHeight && settings.heightDetail == 0
            ? baseHeight : kernels.height.apply(extent:output, arguments:[baseHeight,linearDiffuse,lowPhoto,settings.heightDetail])
        guard let height = heightCandidate,
              let roughness = modelMaps["roughness"]?.image ?? kernels.roughness.apply(extent:output,arguments:[linearDiffuse,lowPhoto,settings.roughnessBase,settings.roughnessDetail]) else {
            throw TextureError.processing("Material map construction failed.")
        }
        // Central difference across two texels; physical width gives resolution-independent slopes.
        let slope = Float(settings.displacementScaleMeters / settings.materialWidthMeters) * Float(settings.outputSize) / 2
        guard let normal = modelMaps["normal"]?.image ?? kernels.normal.apply(extent:output, roiCallback:{ _, rect in rect.insetBy(dx:-2,dy:-2) },
                                               arguments:[height.clampedToExtent(),slope]) else {
            throw TextureError.processing("Normal map construction failed.")
        }
        if let roughness = modelMaps["roughness"] {
            warnings.removeAll { $0.hasPrefix("Roughness is an editable contrast estimate") }
            warnings.append("\(roughness.sourceLabel). Predicted linear roughness values are preserved without heuristic contrast adjustment.")
        }
        if let normal = modelMaps["normal"] {
            warnings.append("\(normal.sourceLabel). Predicted OpenGL normal values are preserved; inspect consistency with the displacement map.")
        }
        return MaterialResult(diffuse:diffuse,roughness:roughness,normal:normal,height:height,crop:crop,
            warnings:warnings,outputSize:settings.outputSize,depthOrigin:depthOrigin,settings:settings,
            sourceURL:source.url,camera:source.camera)
    }

    func preview(_ image: CIImage, maxDimension: Int = 1024) throws -> CGImage {
        let extent = image.extent
        guard extent.width > 0, extent.height > 0, maxDimension > 0, maxDimension <= 2048 else {
            throw TextureError.invalidImage("Invalid preview dimensions.")
        }
        let scale = min(1,Double(maxDimension)/max(extent.width,extent.height))
        let small = image.transformed(by:CGAffineTransform(scaleX:scale,y:scale))
        guard let cg = context.createCGImage(small,from:small.extent,format:.RGBA8,colorSpace:srgb) else {
            throw TextureError.processing("Cannot render this preview.")
        }
        return cg
    }

    /// For linear map previews only: don't bake display gamma into exported map data.
    func mapPreview(_ image: CIImage, maxDimension: Int = 1024) throws -> CGImage {
        let scale = min(1,Double(maxDimension)/max(image.extent.width,image.extent.height))
        let small = image.transformed(by:CGAffineTransform(scaleX:scale,y:scale))
        guard let cg = dataContext.createCGImage(small,from:small.extent,format:.RGBA8,colorSpace:srgb) else {
            throw TextureError.processing("Cannot render this map preview.")
        }
        return cg
    }

    func export(_ result: MaterialResult, to folder: URL, precision: EXRPrecision) throws -> [URL] {
        try Task.checkCancellation()
        try FileManager.default.createDirectory(at:folder,withIntermediateDirectories:true)
        let diskEstimate = Int64(result.outputSize) * Int64(result.outputSize) * 32 + 16 * 1024 * 1024
        if let available = try folder.resourceValues(forKeys:[.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage,
           available < diskEstimate {
            throw TextureError.insufficientResources("This export may need \(diskEstimate/1024/1024) MiB of disk space. Free space or choose a smaller texture size.")
        }
        let staging = folder.appendingPathComponent(".texture-studio-\(UUID().uuidString)",isDirectory:true)
        try FileManager.default.createDirectory(at:staging,withIntermediateDirectories:true)
        defer { try? FileManager.default.removeItem(at:staging) }
        let names = ["diffuse.png","roughness.exr","normal.exr","displacement.exr","material.json","Blender-setup.txt"]
        // A previous export stays intact unless the whole replacement can be staged successfully.
        let diffuseURL = staging.appendingPathComponent(names[0])
        try writeDiffuse(result.diffuse, to: diffuseURL)
        for (name,image,isColor) in [(names[1],result.roughness,false),(names[2],result.normal,true),(names[3],result.height,false)] {
            try Task.checkCancellation()
            let destination = staging.appendingPathComponent(name)
            if precision == .float16 {
                try dataContext.writeOpenEXRRepresentation(of:image,to:destination)
                try FloatEXRWriter.verifyChannelPrecision(at:destination, expected:.float16)
            } else {
                try FloatEXRWriter.write(image, to:destination, context:dataContext, color:isColor)
                try FloatEXRWriter.verifyChannelPrecision(at:destination, expected:.float32)
            }
        }
        struct Manifest: Encodable {
            let schema: Int
            let source: String
            let camera: CameraMetadata
            let settings: TextureSettings
            let depthOrigin: String
            let precision: EXRPrecision
            let croppedSourceRectangle: [Double]
            let maps: [String:String]
            let notes: [String]
        }
        let manifest = Manifest(schema:1,source:result.sourceURL.path,camera:result.camera,settings:result.settings,
            depthOrigin:result.depthOrigin,precision:precision,
            croppedSourceRectangle:[result.crop.minX,result.crop.minY,result.crop.width,result.crop.height],
            maps:["diffuse":"16-bit sRGB PNG", "roughness":"linear grayscale EXR",
                  "normal":"linear RGB EXR, tangent-space OpenGL +Y, encoded [0,1]",
                  "displacement":"linear grayscale EXR, relative height [0,1], neutral 0.5"],notes:result.warnings)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted,.sortedKeys]
        try encoder.encode(manifest).write(to:staging.appendingPathComponent(names[4]))
        let blender = """
        Blender Cycles setup
        Material dimensions: \(result.settings.materialWidthMeters) metres across U and V.
        Diffuse PNG: sRGB → Principled BSDF Base Color.
        Roughness EXR: Non-Color → Principled BSDF Roughness.
        Normal EXR: Non-Color → Normal Map node (Tangent Space, Strength 1) → BSDF Normal.
        Displacement EXR: Non-Color → Displacement node (Midlevel 0.5, Scale \(result.settings.displacementScaleMeters) metres) → Material Output Displacement.
        Enable supported Cycles displacement and provide enough subdivision for the map resolution.
        Normal and displacement describe the same relief. Reduce normal strength when full geometry displacement is enabled to avoid doubling the surface effect.
        Height/roughness are material estimates. Review the output under neutral lighting; no scientific accuracy or seamless tiling is implied.
        """
        try blender.write(to:staging.appendingPathComponent(names[5]),atomically:true,encoding:.utf8)
        for name in names {
            let existing = folder.appendingPathComponent(name)
            guard !FileManager.default.fileExists(atPath:existing.path) else {
                throw TextureError.processing("\(name) already exists in this folder. Choose a new export folder to preserve the previous material.")
            }
        }
        var published = [URL]()
        do {
            try Task.checkCancellation()
            for name in names {
                let destination = folder.appendingPathComponent(name)
                try FileManager.default.moveItem(at:staging.appendingPathComponent(name),to:destination)
                published.append(destination)
            }
        } catch {
            for file in published { try? FileManager.default.removeItem(at:file) }
            throw error
        }
        context.clearCaches(); dataContext.clearCaches()
        return published
    }

    private func validate(_ settings: TextureSettings) throws {
        guard (TextureSettings.outputSizes.contains(settings.outputSize) || settings.outputSize == 4098),
              settings.lensDistortion.isFinite, abs(settings.lensDistortion) <= 0.15,
              [settings.lightingStrength,settings.lightingRadius,settings.heightStrength,
               settings.heightDetail,settings.surfacePlaneRemoval,settings.depthCleanup,settings.roughnessBase,settings.roughnessDetail].allSatisfy({ $0.isFinite && $0 >= 0 && $0 <= 1 }),
              settings.materialWidthMeters.isFinite, settings.materialWidthMeters > 0,
              settings.displacementScaleMeters.isFinite, settings.displacementScaleMeters >= 0 else {
            throw TextureError.invalidSettings("Use a supported output size and finite material controls within their ranges.")
        }
    }

    static func depthCleanupSide(outputSize:Int, depthExtent:CGRect) -> Int {
        min(outputSize,max(64,Int(ceil(max(depthExtent.width,depthExtent.height)))))
    }

    /// A snapshot of free/reclaimable VM pages, refreshed for every render.
    static func availableMemoryBytes() -> UInt64? {
        var statistics = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to:&statistics) { pointer in
            pointer.withMemoryRebound(to:integer_t.self,capacity:Int(count)) {
                host_statistics64(mach_host_self(),HOST_VM_INFO64,$0,&count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        // Purgeable pages may already be counted as inactive; do not count them twice.
        let pages = UInt64(statistics.free_count) + UInt64(statistics.speculative_count) +
            max(UInt64(statistics.inactive_count),UInt64(statistics.purgeable_count))
        return pages * UInt64(max(1,sysconf(_SC_PAGESIZE)))
    }

    private func supportingIndices(_ source: CGImageSource, primaryIndex: Int) -> [Int] {
        let properties = CGImageSourceCopyProperties(source,nil) as? [String:Any] ?? [:]
        let perImage = (0..<CGImageSourceGetCount(source)).map {
            CGImageSourceCopyPropertiesAtIndex(source,$0,nil) as? [String:Any] ?? [:]
        }
        return Self.supportingImageIndices(globalProperties:properties,perImageProperties:perImage,primaryIndex:primaryIndex)
    }

    /// Only documented stereo groups count as companion photographs, including the
    /// FileContents container and per-image group representations used by ImageIO.
    static func supportingImageIndices(globalProperties: [String:Any], perImageProperties: [[String:Any]],
                                       primaryIndex: Int) -> [Int] {
        func groups(_ value: Any?) -> [[String:Any]] {
            if let array = value as? [[String:Any]] { return array }
            if let dictionary = value as? [String:Any] { return [dictionary] }
            return []
        }
        let contents = globalProperties[kCGImagePropertyFileContentsDictionary as String] as? [String:Any] ?? [:]
        let globalGroups = groups(globalProperties[kCGImagePropertyGroups as String]) + groups(contents[kCGImagePropertyGroups as String])
        let imageGroups = perImageProperties.map { groups($0[kCGImagePropertyGroups as String]) }
        guard imageGroups.indices.contains(primaryIndex) else { return [] }
        let type = kCGImagePropertyGroupType as String
        let stereo = kCGImagePropertyGroupTypeStereoPair as String
        let groupIndex = kCGImagePropertyGroupIndex as String
        let primaryGroups = imageGroups[primaryIndex].filter { ($0[type] as? String) == stereo }
        let primaryGroupIDs = Set(primaryGroups.compactMap { ($0[groupIndex] as? NSNumber)?.intValue })
        var indices = Set<Int>()
        for group in globalGroups + primaryGroups {
            guard (group[type] as? String) == stereo else { continue }
            let pair = [kCGImagePropertyGroupImageIndexLeft,kCGImagePropertyGroupImageIndexRight,
                        kCGImagePropertyGroupImageIndexMonoscopic].compactMap { (group[$0 as String] as? NSNumber)?.intValue }
            guard pair.contains(primaryIndex) else { continue }
            for index in pair where index != primaryIndex && perImageProperties.indices.contains(index) {
                indices.insert(index)
            }
        }
        let roles = [kCGImagePropertyGroupImageIsLeftImage,kCGImagePropertyGroupImageIsRightImage,
                     kCGImagePropertyGroupImageIsMonoscopicImage]
        for (index,groups) in imageGroups.enumerated() where index != primaryIndex {
            if groups.contains(where:{ group in
                guard (group[type] as? String) == stereo,
                      let id = (group[groupIndex] as? NSNumber)?.intValue,
                      primaryGroupIDs.contains(id) else { return false }
                return roles.contains { (group[$0 as String] as? NSNumber)?.boolValue == true }
            }) { indices.insert(index) }
        }
        // Never treat arbitrary HEIF items, gain maps, thumbnails, or mattes as photographs.
        return indices.sorted().prefix(2).map { $0 }
    }

    private func lensCorrect(_ image: CIImage, amount: Double, kernels: Kernels) throws -> CIImage {
        guard amount != 0 else { return image }
        let extent = image.extent
        let centre = CIVector(x:extent.midX,y:extent.midY)
        let halfExtent = CIVector(x:extent.width/2,y:extent.height/2)
        let zoom = 1 + max(0,amount*2) + 0.005
        guard let corrected = kernels.lens.apply(extent:extent,roiCallback:{ _, _ in extent },image:image,
                                                 arguments:[centre,halfExtent,Float(amount),Float(zoom)]) else {
            throw TextureError.processing("Lens correction failed.")
        }
        return corrected
    }

    private func transform(_ image: CIImage, corners: [CGPoint], crop: CGRect, size: Int) -> CIImage {
        let perspective = image.applyingFilter("CIPerspectiveTransform",parameters:[
            "inputBottomLeft":CIVector(cgPoint:corners[0]),"inputBottomRight":CIVector(cgPoint:corners[1]),
            "inputTopRight":CIVector(cgPoint:corners[2]),"inputTopLeft":CIVector(cgPoint:corners[3])])
        let cropped = perspective.cropped(to:crop).transformed(by:CGAffineTransform(translationX:-crop.minX,y:-crop.minY))
        return resize(cropped,to:CGSize(width:size,height:size))
    }

    private func resize(_ image: CIImage, to size: CGSize) -> CIImage {
        let zero = image.transformed(by:CGAffineTransform(translationX:-image.extent.minX,y:-image.extent.minY))
        // A finite bitmap's border texel is centred half a pixel inside its extent.
        // Upsampling without clamping blends that texel with transparent zero;
        // distance-to-height normalization then turns it into a false raised rim.
        let scale=CGAffineTransform(scaleX:size.width/zero.extent.width,y:size.height/zero.extent.height)
        return zero.clampedToExtent().transformed(by:scale)
            .cropped(to:CGRect(origin:.zero,size:size))
    }

    private func meanLuminance(_ image: CIImage) throws -> Float {
        let average = image.applyingFilter("CIAreaAverage",parameters:["inputExtent":CIVector(cgRect:image.extent)])
        var sample = [Float](repeating:0,count:4)
        sample.withUnsafeMutableBytes { context.render(average,toBitmap:$0.baseAddress!,rowBytes:16,
            bounds:CGRect(x:0,y:0,width:1,height:1),format:.RGBAf,colorSpace:linear) }
        // Core Image averages premultiplied color. Normalize by visible coverage
        // so transparent holes do not darken the material's lighting target.
        guard sample[3].isFinite, sample[3] > 1e-6 else { return 0.18 }
        let mean = (sample[0]*0.2126+sample[1]*0.7152+sample[2]*0.0722) / sample[3]
        return mean.isFinite ? max(0.005,mean) : 0.18
    }

    private struct Kernels {
        let delight: CIColorKernel
        let height: CIColorKernel
        let roughness: CIColorKernel
        let normal: CIKernel
        let lens: CIWarpKernel
    }
    private func loadKernels() throws -> Kernels {
        if let kernels { return kernels }
        guard let url = libraryURL ?? Bundle.main.url(forResource:"TextureKernels",withExtension:"metallib") else {
            throw TextureError.missingKernels
        }
        let data = try Data(contentsOf:url)
        let loaded = try Kernels(delight:CIColorKernel(functionName:"textureDelight",fromMetalLibraryData:data),
            height:CIColorKernel(functionName:"textureHeight",fromMetalLibraryData:data),
            roughness:CIColorKernel(functionName:"textureRoughness",fromMetalLibraryData:data),
            normal:CIKernel(functionName:"textureNormal",fromMetalLibraryData:data),
            lens:CIWarpKernel(functionName:"textureLens",fromMetalLibraryData:data))
        kernels = loaded
        return loaded
    }
}

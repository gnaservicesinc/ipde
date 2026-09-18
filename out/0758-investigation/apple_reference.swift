import Foundation
import ImageIO
import AVFoundation
import CoreVideo
let url = URL(fileURLWithPath: "/Users/andrewsmith/Downloads/IMG_0758.HEIC")
let src = CGImageSourceCreateWithURL(url as CFURL, nil)!
let dict = CGImageSourceCopyAuxiliaryDataInfoAtIndex(src, 0, kCGImageAuxiliaryDataTypeDisparity)!
let depth = try AVDepthData(fromDictionaryRepresentation: dict as! [AnyHashable: Any])
print("accuracy", depth.depthDataAccuracy.rawValue, "filtered", depth.isDepthDataFiltered)
if let cal = depth.cameraCalibrationData {print("calibration",cal.intrinsicMatrix,cal.intrinsicMatrixReferenceDimensions)}
let map = depth.converting(toDepthDataType: kCVPixelFormatType_DepthFloat32).depthDataMap
CVPixelBufferLockBaseAddress(map, .readOnly)
let w = CVPixelBufferGetWidth(map), h = CVPixelBufferGetHeight(map), stride = CVPixelBufferGetBytesPerRow(map)
print("dimensions",w,h,"stride",stride)
let ptr = CVPixelBufferGetBaseAddress(map)!
var data = Data()
for y in 0..<h { data.append(ptr.advanced(by:y*stride).assumingMemoryBound(to:UInt8.self),count:w*4) }
try data.write(to:URL(fileURLWithPath:"out/0758-investigation/apple_native_depth.f32"))
CVPixelBufferUnlockBaseAddress(map, .readOnly)

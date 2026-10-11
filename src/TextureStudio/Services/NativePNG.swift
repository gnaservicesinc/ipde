import Foundation
import zlib

/// Integer PNG storage. This decoder never enters a color-managed image
/// pipeline: filters, compression and Adam7 are reversed directly on bytes.
struct NativePNG: Sendable {
    struct Header: Equatable, Sendable {
        let width: Int
        let height: Int
        let bits: Int
        let channels: Int
        let color: UInt8
        let interlace: UInt8
        var bytesPerPixel: Int { channels * bits / 8 }
    }
    let header: Header
    var pixels: Data
    let colorChunks: [(String, Data)]
    private static let signature = Data([137, 80, 78, 71, 13, 10, 26, 10])
    private static let colorTypes: [UInt8: Int] = [0: 1, 2: 3, 4: 2, 6: 4]
    private static let retainedChunks: Set<String> = ["cHRM", "gAMA", "iCCP", "sRGB", "sBIT", "cICP", "mDCV", "cLLI", "tRNS"]

    static func inspect(_ url: URL) throws -> Header {
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        return try parseHeader(file.read(upToCount: 33) ?? Data())
    }
    private static func parseHeader(_ data: Data) throws -> Header {
        guard data.count >= 33, data.prefix(8) == signature, integer(data, 8) == 13,
              data.subdata(in: 12..<16) == Data("IHDR".utf8),
              checksum(data.subdata(in: 12..<29)) == integer(data, 29) else { throw invalid("Invalid PNG IHDR or checksum.") }
        let width = Int(integer(data, 16)), height = Int(integer(data, 20)), bits = Int(data[24]), color = data[25]
        guard width > 0, height > 0, width <= 65536, height <= 65536,
              width * height <= 150_000_000, [8, 16].contains(bits), let channels = colorTypes[color],
              data[26] == 0, data[27] == 0, data[28] <= 1 else { throw invalid("Unsupported native PNG storage.") }
        return Header(width: width, height: height, bits: bits, channels: channels, color: color, interlace: data[28])
    }
    static func decode(_ data: Data) throws -> NativePNG {
        let header = try parseHeader(data)
        var cursor = 8, compressed = Data(), colors: [(String, Data)] = [], ended = false
        var transparencySeen = false, compressedSeen = false
        try data.withUnsafeBytes { storage in
            let bytes = storage.bindMemory(to: UInt8.self)
            while cursor + 12 <= bytes.count {
                try Task.checkCancellation()
                let length = Int(integer(bytes, cursor))
                guard length <= bytes.count - cursor - 12 else { throw invalid("Truncated PNG chunk.") }
                let kind = String(decoding: UnsafeBufferPointer(start: bytes.baseAddress! + cursor + 4, count: 4), as: UTF8.self)
                guard checksum(bytes, offset: cursor + 4, count: length + 4) == integer(bytes, cursor + 8 + length) else { throw invalid("PNG chunk checksum changed.") }
                if kind == "IHDR", cursor != 8 { throw invalid("PNG contains more than one image header.") }
                if kind == "tRNS" {
                    guard !transparencySeen, !compressedSeen else { throw invalid("Invalid PNG transparency chunk order.") }
                    try validateTransparency(Data(bytes: bytes.baseAddress! + cursor + 8, count: length), header: header)
                    transparencySeen = true
                }
                if kind == "IDAT" { compressedSeen = true; compressed.append(bytes.baseAddress! + cursor + 8, count: length) }
                else if retainedChunks.contains(kind) { colors.append((kind, Data(bytes: bytes.baseAddress! + cursor + 8, count: length))) }
                else if kind == "IEND" { ended = true; cursor += length + 12; break }
                else if kind != "IHDR", bytes[cursor + 4] & 32 == 0 { throw invalid("Unsupported critical PNG chunk.") }
                cursor += length + 12
            }
        }
        guard ended, cursor == data.count, !compressed.isEmpty else { throw invalid("PNG image is incomplete.") }
        let passes = header.interlace == 0 ? [(0, 0, 1, 1)] : [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
        let bytesPerPixel = header.bytesPerPixel
        let layouts = passes.map { x, y, dx, dy in
            (x, y, dx, dy, max(0, (header.width - x + dx - 1) / dx), max(0, (header.height - y + dy - 1) / dy))
        }
        let expected = layouts.reduce(0) { $0 + ($1.4 > 0 && $1.5 > 0 ? ($1.4 * bytesPerPixel + 1) * $1.5 : 0) }
        var filtered = Data(count: expected), actual = uLongf(expected)
        let status = filtered.withUnsafeMutableBytes { destination in
            compressed.withUnsafeBytes { source in
                uncompress(destination.bindMemory(to: Bytef.self).baseAddress!, &actual,
                    source.bindMemory(to: Bytef.self).baseAddress!, uLong(compressed.count))
            }
        }
        guard status == Z_OK, actual == expected else { throw invalid("PNG pixel stream has an invalid size.") }
        var pixels = Data(count: header.width * header.height * bytesPerPixel), offset = 0
        try filtered.withUnsafeMutableBytes { filteredStorage in
            try pixels.withUnsafeMutableBytes { pixelStorage in
                let scanlines = filteredStorage.bindMemory(to: UInt8.self).baseAddress!
                let destination = pixelStorage.bindMemory(to: UInt8.self).baseAddress!
                for (x, y, dx, dy, width, height) in layouts where width > 0 && height > 0 {
                    let rowBytes = width * bytesPerPixel
                    var previous: UnsafeMutablePointer<UInt8>?
                    for rowIndex in 0..<height {
                        try Task.checkCancellation()
                        let filter = scanlines[offset]; offset += 1
                        guard filter <= 4 else { throw invalid("Unknown PNG row filter.") }
                        let row = scanlines + offset; offset += rowBytes
                        switch filter {
                        case 0: break
                        case 1:
                            for column in bytesPerPixel..<rowBytes { row[column] &+= row[column - bytesPerPixel] }
                        case 2:
                            if let previous { for column in 0..<rowBytes { row[column] &+= previous[column] } }
                        case 3:
                            for column in 0..<bytesPerPixel { row[column] &+= (previous?[column] ?? 0) / 2 }
                            for column in bytesPerPixel..<rowBytes { row[column] &+= UInt8((Int(row[column - bytesPerPixel]) + Int(previous?[column] ?? 0)) / 2) }
                        default:
                            if let previous {
                                for column in 0..<bytesPerPixel { row[column] &+= previous[column] }
                                for column in bytesPerPixel..<rowBytes { row[column] &+= paeth(row[column - bytesPerPixel], previous[column], previous[column - bytesPerPixel]) }
                            } else { for column in bytesPerPixel..<rowBytes { row[column] &+= row[column - bytesPerPixel] } }
                        }
                        let target = destination + ((y + rowIndex * dy) * header.width + x) * bytesPerPixel
                        if dx == 1 { target.update(from: row, count: rowBytes) }
                        else {
                            for column in 0..<width { (target + column * dx * bytesPerPixel).update(from: row + column * bytesPerPixel, count: bytesPerPixel) }
                        }
                        previous = row
                    }
                }
            }
        }
        return NativePNG(header: header, pixels: pixels, colorChunks: colors)
    }
    static func sourceMetadata(_ data: Data) throws -> [String: Any] {
        let header = try parseHeader(data)
        var metadata: [String: Any] = ["width": header.width, "height": header.height, "sample_bits": header.bits, "channels": header.channels, "png_color_type": Int(header.color), "interlace": Int(header.interlace)]
        var cursor = 8, ended = false, transparencySeen = false, compressedSeen = false
        try data.withUnsafeBytes { storage in
            let bytes = storage.bindMemory(to: UInt8.self)
            while cursor + 12 <= bytes.count {
                try Task.checkCancellation()
                let length = Int(integer(bytes, cursor))
                guard length <= bytes.count - cursor - 12 else { throw invalid("Truncated PNG chunk.") }
                let kind = String(decoding: UnsafeBufferPointer(start: bytes.baseAddress! + cursor + 4, count: 4), as: UTF8.self)
                guard checksum(bytes, offset: cursor + 4, count: length + 4) == integer(bytes, cursor + 8 + length) else { throw invalid("PNG chunk checksum changed.") }
                if kind == "IHDR", cursor != 8 { throw invalid("PNG contains more than one image header.") }
                if kind == "tRNS" {
                    guard !transparencySeen, !compressedSeen else { throw invalid("Invalid PNG transparency chunk order.") }
                    try validateTransparency(Data(bytes: bytes.baseAddress! + cursor + 8, count: length), header: header)
                    transparencySeen = true
                }
                if kind == "IDAT" { compressedSeen = true }
                if retainedChunks.contains(kind) {
                    if kind == "gAMA", length == 4 { metadata["png_gamma"] = Double(integer(bytes, cursor + 8)) / 100000 }
                    if kind == "sRGB", length == 1 { metadata["srgb_rendering_intent"] = Int(bytes[cursor + 8]) }
                }
                cursor += length + 12
                if kind == "IEND" { ended = true; break }
            }
        }
        guard ended, cursor == data.count else { throw invalid("PNG image is incomplete.") }
        // Profile bytes stay in their PNG. They are read only when exporting a
        // crop, never expanded into text and repeated in dataset manifests.
        return metadata
    }
    /// Read a source once while retaining only two scanlines and the selected
    /// crop pixels. Original resolution never determines a full-image buffer.
    static func crop(_ url: URL, rectangle: [Int], flipGreen: Bool = false) throws -> NativePNG {
        try crops(url, rectangles: [rectangle], flipGreen: flipGreen)[0]
    }
    static func crop(_ data: Data, rectangle: [Int], flipGreen: Bool = false) throws -> NativePNG {
        try crops(data, rectangles: [rectangle], flipGreen: flipGreen)[0]
    }
    static func crops(_ url: URL, rectangles: [[Int]], flipGreen: Bool = false) throws -> [NativePNG] {
        try crops(Data(contentsOf: url, options: .mappedIfSafe), rectangles: rectangles, flipGreen: flipGreen)
    }
    static func crops(_ data: Data, rectangles: [[Int]], flipGreen: Bool = false) throws -> [NativePNG] {
        let header = try parseHeader(data), bpp = header.bytesPerPixel
        guard !rectangles.isEmpty, !flipGreen || [3, 4].contains(header.channels) else { throw invalid("Native PNG crop channels are invalid.") }
        for r in rectangles {
            guard r.count == 4, r[0] >= 0, r[1] >= 0, r[2] > 0, r[3] > 0,
                  r[2] <= header.width, r[3] <= header.height,
                  r[0] <= header.width - r[2], r[1] <= header.height - r[3] else { throw invalid("Crop is outside the original PNG grid.") }
        }
        var selected = rectangles.map { Data(count: $0[2] * $0[3] * bpp) }
        let passes = header.interlace == 0 ? [(0, 0, 1, 1)] : [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
        let layouts = passes.map { x, y, dx, dy in
            (x, y, dx, dy, max(0, (header.width - x + dx - 1) / dx), max(0, (header.height - y + dy - 1) / dy))
        }.filter { $0.4 > 0 && $0.5 > 0 }
        var pass = 0, rowIndex = 0, filled = 0
        var row = [UInt8](repeating: 0, count: header.width * bpp + 1)
        var previous = [UInt8](repeating: 0, count: header.width * bpp)
        func finishRow() throws {
            let (x, y, dx, dy, width, height) = layouts[pass], rowBytes = width * bpp
            guard row[0] <= 4 else { throw invalid("Unknown PNG row filter.") }
            for column in 0..<rowBytes {
                let left = column >= bpp ? row[column + 1 - bpp] : 0
                let up = previous[column], upperLeft = column >= bpp ? previous[column - bpp] : 0
                switch row[0] {
                case 0: break
                case 1: row[column + 1] &+= left
                case 2: row[column + 1] &+= up
                case 3: row[column + 1] &+= UInt8((Int(left) + Int(up)) / 2)
                default: row[column + 1] &+= paeth(left, up, upperLeft)
                }
            }
            let sourceY = y + rowIndex * dy
            row.withUnsafeBufferPointer { bytes in
                for index in rectangles.indices {
                    let r = rectangles[index]
                    guard sourceY >= r[1], sourceY < r[1] + r[3] else { continue }
                    selected[index].withUnsafeMutableBytes { output in
                        let destination = output.bindMemory(to: UInt8.self).baseAddress! + (sourceY - r[1]) * r[2] * bpp
                        if dx == 1 {
                            destination.update(from: bytes.baseAddress! + 1 + r[0] * bpp, count: r[2] * bpp)
                        } else {
                            for column in 0..<width {
                                let sourceX = x + column * dx
                                if sourceX >= r[0], sourceX < r[0] + r[2] {
                                    (destination + (sourceX - r[0]) * bpp).update(from: bytes.baseAddress! + 1 + column * bpp, count: bpp)
                                }
                            }
                        }
                    }
                }
            }
            previous.replaceSubrange(0..<rowBytes, with: row[1...rowBytes])
            rowIndex += 1; filled = 0
            if rowIndex == height {
                pass += 1; rowIndex = 0
                previous.withUnsafeMutableBufferPointer { $0.initialize(repeating: 0) }
            }
        }
        var stream = z_stream()
        guard inflateInit_(&stream, ZLIB_VERSION, Int32(MemoryLayout<z_stream>.size)) == Z_OK else { throw invalid("PNG decompression could not start.") }
        defer { inflateEnd(&stream) }
        var cursor = 8, colors: [(String, Data)] = [], ended = false, streamEnded = false, compressedSeen = false, profileBytes = 0
        var transparencySeen = false
        try data.withUnsafeBytes { storage in
            let bytes = storage.bindMemory(to: UInt8.self)
            while cursor + 12 <= bytes.count {
                try Task.checkCancellation()
                let length = Int(integer(bytes, cursor))
                guard length <= bytes.count - cursor - 12 else { throw invalid("Truncated PNG chunk.") }
                let kind = String(decoding: UnsafeBufferPointer(start: bytes.baseAddress! + cursor + 4, count: 4), as: UTF8.self)
                guard checksum(bytes, offset: cursor + 4, count: length + 4) == integer(bytes, cursor + 8 + length) else { throw invalid("PNG chunk checksum changed.") }
                if kind == "IHDR", cursor != 8 { throw invalid("PNG contains more than one image header.") }
                if kind == "tRNS" {
                    guard !transparencySeen, !compressedSeen else { throw invalid("Invalid PNG transparency chunk order.") }
                    try validateTransparency(Data(bytes: bytes.baseAddress! + cursor + 8, count: length), header: header)
                    transparencySeen = true
                }
                if retainedChunks.contains(kind) {
                    profileBytes += length
                    guard profileBytes <= 16 * 1024 * 1024 else { throw invalid("PNG color metadata exceeds the native crop budget.") }
                    colors.append((kind, Data(bytes: bytes.baseAddress! + cursor + 8, count: length)))
                } else if kind == "IDAT" {
                    guard !streamEnded || length == 0 else { throw invalid("PNG contains trailing compressed pixels.") }
                    compressedSeen = true
                    stream.next_in = UnsafeMutablePointer(mutating: bytes.baseAddress! + cursor + 8)
                    stream.avail_in = uInt(length)
                    while stream.avail_in > 0 {
                        try Task.checkCancellation()
                        let expected = pass < layouts.count ? layouts[pass].4 * bpp + 1 : 1
                        let availableBefore = stream.avail_in
                        let status = row.withUnsafeMutableBufferPointer { buffer -> Int32 in
                            stream.next_out = buffer.baseAddress! + filled
                            stream.avail_out = uInt(expected - filled)
                            return inflate(&stream, Z_NO_FLUSH)
                        }
                        let produced = expected - filled - Int(stream.avail_out)
                        if pass >= layouts.count, produced > 0 { throw invalid("PNG pixel stream has an invalid size.") }
                        filled += produced
                        if pass < layouts.count, filled == expected { try finishRow() }
                        guard status == Z_OK || status == Z_STREAM_END else { throw invalid("PNG pixel stream has an invalid size.") }
                        if status == Z_STREAM_END {
                            streamEnded = true
                            guard stream.avail_in == 0, pass == layouts.count, filled == 0 else { throw invalid("PNG pixel stream has an invalid size.") }
                            break
                        }
                        guard produced > 0 || stream.avail_in < availableBefore else { throw invalid("PNG decompression stopped before its pixels were complete.") }
                    }
                } else if kind == "IEND" { ended = true; cursor += length + 12; break }
                else if kind != "IHDR", bytes[cursor + 4] & 32 == 0 { throw invalid("Unsupported critical PNG chunk.") }
                cursor += length + 12
            }
        }
        guard ended, cursor == data.count, compressedSeen, streamEnded, pass == layouts.count else { throw invalid("PNG image is incomplete.") }
        return rectangles.enumerated().map { index, r in
            if flipGreen {
                selected[index].withUnsafeMutableBytes { storage in
                    let bytes = storage.bindMemory(to: UInt8.self)
                    for pixel in 0..<r[2] * r[3] { for byte in 0..<header.bits / 8 { bytes[pixel * bpp + header.bits / 8 + byte] ^= 255 } }
                }
            }
            return NativePNG(header: Header(width: r[2], height: r[3], bits: header.bits, channels: header.channels, color: header.color, interlace: 0), pixels: selected[index], colorChunks: transformedChunks(colors, header: header, flipGreen: flipGreen))
        }
    }
    func crop(_ rectangle: [Int], flipGreen: Bool = false) throws -> NativePNG {
        try validateStorage()
        guard rectangle.count == 4, rectangle[0] >= 0, rectangle[1] >= 0,
              rectangle[2] > 0, rectangle[3] > 0, rectangle[2] <= header.width,
              rectangle[3] <= header.height, rectangle[0] <= header.width - rectangle[2],
              rectangle[1] <= header.height - rectangle[3] else { throw Self.invalid("Crop is outside the original PNG grid.") }
        guard !flipGreen || [3, 4].contains(header.channels) else { throw Self.invalid("DirectX normal maps need RGB integer channels.") }
        let width = rectangle[2], height = rectangle[3], rowBytes = width * header.bytesPerPixel
        let componentBytes = header.bits / 8, bytesPerPixel = header.bytesPerPixel
        var selected = Data(count: rowBytes * height)
        try pixels.withUnsafeBytes { source in
            try selected.withUnsafeMutableBytes { destination in
                let sourceBytes = source.bindMemory(to: UInt8.self).baseAddress!
                let selectedBytes = destination.bindMemory(to: UInt8.self).baseAddress!
                for row in 0..<height {
                    try Task.checkCancellation()
                    let start = ((rectangle[1] + row) * header.width + rectangle[0]) * bytesPerPixel
                    let target = selectedBytes + row * rowBytes
                    target.update(from: sourceBytes + start, count: rowBytes)
                    if flipGreen {
                        // max - unsigned code is a byte complement at either precision.
                        for pixel in 0..<width { for byte in 0..<componentBytes { target[pixel * bytesPerPixel + componentBytes + byte] ^= 255 } }
                    }
                }
            }
        }
        return NativePNG(header: Header(width: width, height: height, bits: header.bits,
            channels: header.channels, color: header.color, interlace: 0), pixels: selected,
            colorChunks: Self.transformedChunks(colorChunks, header: header, flipGreen: flipGreen))
    }
    func encoded() throws -> Data {
        try validateStorage()
        let rowBytes = header.width * header.bytesPerPixel
        var scanlines = Data(count: pixels.count + header.height)
        try pixels.withUnsafeBytes { source in
            try scanlines.withUnsafeMutableBytes { destination in
                let sourceBytes = source.bindMemory(to: UInt8.self).baseAddress!
                let rows = destination.bindMemory(to: UInt8.self).baseAddress!
                for row in 0..<header.height {
                    try Task.checkCancellation()
                    let encoded = rows + row * (rowBytes + 1)
                    let raw = sourceBytes + row * rowBytes
                    encoded[0] = 1
                    (encoded + 1).update(from: raw, count: header.bytesPerPixel)
                    for column in header.bytesPerPixel..<rowBytes { encoded[column + 1] = raw[column] &- raw[column - header.bytesPerPixel] }
                }
            }
        }
        try Task.checkCancellation()
        var length = compressBound(uLong(scanlines.count)), compressed = Data(count: Int(length))
        let status = compressed.withUnsafeMutableBytes { destination in
            scanlines.withUnsafeBytes { source in
                compress2(destination.bindMemory(to: Bytef.self).baseAddress!, &length,
                    source.bindMemory(to: Bytef.self).baseAddress!, uLong(scanlines.count), 1)
            }
        }
        guard status == Z_OK else { throw Self.invalid("Lossless PNG encoding failed.") }
        try Task.checkCancellation()
        compressed.count = Int(length)
        var ihdr = Data(); ihdr.appendBE(UInt32(header.width)); ihdr.appendBE(UInt32(header.height))
        ihdr.append(contentsOf: [UInt8(header.bits), header.color, 0, 0, 0])
        var output = Self.signature
        output.append(Self.chunk("IHDR", ihdr))
        for (kind, payload) in colorChunks { output.append(Self.chunk(kind, payload)) }
        output.append(Self.chunk("IDAT", compressed)); output.append(Self.chunk("IEND", Data()))
        return output
    }
    /// Model tensors are a separate, explicit conversion from immutable integer
    /// source storage. Returned values use planar CHW order, matching training.
    /// PNG color/data samples are unassociated with alpha. Training pairs are
    /// already planar: alpha and color-key transparency are display metadata,
    /// never a crop/coverage test, compositing operation, or reason to fill pixels.
    func modelFloatSamples(role: String, encoding: String = "linear_data", normalConvention: String = "opengl") throws -> [Float] {
        try validateStorage()
        guard ["input", "height", "roughness", "normal"].contains(role),
              role != "height" || header.bits == 16,
              !["input", "normal"].contains(role) || [3, 4].contains(header.channels) else { throw Self.invalid("The native map channels or precision do not support this training target.") }
        let count = header.width * header.height, outputChannels = ["input", "normal"].contains(role) ? 3 : 1
        let maximum = header.bits == 16 ? 65535 : 255
        let bytesPerPixel = header.bytesPerPixel, componentBytes = header.bits / 8, channels = header.channels
        var result = [Float](repeating: 0, count: count * outputChannels)
        let linear = ["linear", "linear_rgb", "linear_color", "linear_light"].contains(encoding)
        if role == "input", !linear, !["srgb", "sRGB", "source_srgb_assumed", "srgb_display", "srgb_color"].contains(encoding) { throw Self.invalid("Diffuse color transfer must be explicit.") }
        let scalarRGB = ["height", "roughness"].contains(role) && channels >= 3
        let flipGreen = role == "normal" && normalConvention.lowercased() == "directx", linearInput = role == "input" && linear
        try pixels.withUnsafeBytes { storage in
            let bytes = storage.bindMemory(to: UInt8.self).baseAddress!
            func code(_ offset: Int, _ channel: Int) -> Int {
                let index = offset + channel * componentBytes
                return componentBytes == 2 ? Int(bytes[index]) * 256 + Int(bytes[index + 1]) : Int(bytes[index])
            }
            try result.withUnsafeMutableBufferPointer { output in
                let destination = output.baseAddress!
                for pixel in 0..<count {
                    if pixel & 4095 == 0 { try Task.checkCancellation() }
                    let offset = pixel * bytesPerPixel
                    if scalarRGB, (code(offset, 0) != code(offset, 1) || code(offset, 0) != code(offset, 2)) { throw Self.invalid("Scalar material maps need identical RGB codes.") }
                    for channel in 0..<outputChannels {
                        let integer = code(offset, channel)
                        var value = Float(flipGreen && channel == 1 ? maximum - integer : integer) / Float(maximum)
                        if linearInput { value = value <= 0.0031308 ? 12.92 * value : 1.055 * pow(value, 1 / 2.4) - 0.055 }
                        destination[channel * count + pixel] = value
                    }
                }
            }
        }
        return result
    }
    private func validateStorage() throws {
        guard header.width > 0, header.height > 0, header.width <= 65536, header.height <= 65536,
              header.width * header.height <= 150_000_000, [8, 16].contains(header.bits),
              Self.colorTypes[header.color] == header.channels, header.interlace <= 1,
              pixels.count == header.width * header.height * header.bytesPerPixel else { throw Self.invalid("PNG sample storage has an invalid size.") }
        let transparency = colorChunks.filter { $0.0 == "tRNS" }
        guard transparency.count <= 1 else { throw Self.invalid("PNG contains more than one transparency chunk.") }
        if let chunk = transparency.first { try Self.validateTransparency(chunk.1, header: header) }
    }
    private static func validateTransparency(_ data: Data, header: Header) throws {
        // Indexed PNGs are outside native integer storage. For supported gray/RGB
        // images tRNS retains a 16-bit color key without changing their channels.
        let count = header.color == 0 ? 1 : header.color == 2 ? 3 : 0
        guard count > 0, data.count == count * 2 else { throw invalid("Invalid PNG transparency key.") }
        let maximum = header.bits == 16 ? 65535 : 255
        for channel in 0..<count {
            guard Int(data[channel * 2]) * 256 + Int(data[channel * 2 + 1]) <= maximum else {
                throw invalid("PNG transparency key exceeds its sample precision.")
            }
        }
    }
    private static func transformedChunks(_ chunks: [(String, Data)], header: Header, flipGreen: Bool) -> [(String, Data)] {
        guard flipGreen else { return chunks }
        return chunks.map { kind, data in
            guard kind == "tRNS", header.color == 2 else { return (kind, data) }
            var key = data
            let green = Int(key[2]) * 256 + Int(key[3])
            let complemented = (header.bits == 16 ? 65535 : 255) - green
            key[2] = UInt8(complemented >> 8); key[3] = UInt8(truncatingIfNeeded: complemented)
            return (kind, key)
        }
    }
    private static func chunk(_ kind: String, _ payload: Data) -> Data {
        var result = Data(); result.appendBE(UInt32(payload.count))
        let content = Data(kind.utf8) + payload
        result.append(content); result.appendBE(checksum(content)); return result
    }
    private static func checksum(_ data: Data) -> UInt32 {
        data.withUnsafeBytes { UInt32(crc32(0, $0.bindMemory(to: Bytef.self).baseAddress, uInt(data.count))) }
    }
    private static func checksum(_ bytes: UnsafeBufferPointer<UInt8>, offset: Int, count: Int) -> UInt32 {
        UInt32(crc32(0, bytes.baseAddress! + offset, uInt(count)))
    }
    private static func integer(_ data: Data, _ offset: Int) -> UInt32 {
        data[offset..<offset + 4].reduce(0) { ($0 << 8) | UInt32($1) }
    }
    private static func integer(_ bytes: UnsafeBufferPointer<UInt8>, _ offset: Int) -> UInt32 {
        UInt32(bytes[offset]) << 24 | UInt32(bytes[offset + 1]) << 16 | UInt32(bytes[offset + 2]) << 8 | UInt32(bytes[offset + 3])
    }
    private static func paeth(_ a: UInt8, _ b: UInt8, _ c: UInt8) -> UInt8 {
        let p = Int(a) + Int(b) - Int(c), pa = abs(p - Int(a)), pb = abs(p - Int(b)), pc = abs(p - Int(c))
        return pa <= pb && pa <= pc ? a : pb <= pc ? b : c
    }
    private static func invalid(_ message: String) -> StudioError { StudioError(message) }
}

private extension Data {
    mutating func appendBE(_ value: UInt32) { append(contentsOf: [UInt8(truncatingIfNeeded: value >> 24), UInt8(truncatingIfNeeded: value >> 16), UInt8(truncatingIfNeeded: value >> 8), UInt8(truncatingIfNeeded: value)]) }
}

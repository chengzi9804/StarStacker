import CoreGraphics
import Metal
import UIKit

enum StackError: Error {
    case deviceNotFound
    case textureCreationFailed
    case pipelineCreationFailed
    case imageSizeMismatch
    case shaderNotFound(String)
}

final class StackEngine {
    private let device: MTLDevice
    private let commandQueue: MTLCommandQueue
    private let library: MTLLibrary

    /// Downscale factor for alignment search
    private let alignScale = 8

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            throw StackError.deviceNotFound
        }
        device = dev
        guard let queue = device.makeCommandQueue() else {
            throw StackError.deviceNotFound
        }
        commandQueue = queue
        library = try device.makeDefaultLibrary(bundle: .main)
    }

    // MARK: - Public entry

    func stack(
        images: [CGImage],
        sigma: Float,
        align: Bool,
        onProgress: @escaping (Double) -> Void
    ) async throws -> CGImage {
        guard let first = images.first else {
            throw StackError.imageSizeMismatch
        }

        let width = first.width
        let height = first.height

        for img in images where img.width != width || img.height != height {
            throw StackError.imageSizeMismatch
        }

        // 1. Convert all images to Metal textures
        var textures: [MTLTexture] = []
        for (i, cg) in images.enumerated() {
            let tex = try makeTexture(from: cg)
            textures.append(tex)
            onProgress(Double(i + 1) / Double(images.count) * 0.2)
        }

        // 2. Compute alignment shifts (downscaled for speed)
        var shifts: [SIMD2<Float>] = .init(repeating: .zero, count: textures.count)
        if align && textures.count > 1 {
            shifts = try await computeShifts(
                reference: textures[0],
                others: Array(textures.dropFirst())
            )
            onProgress(0.45)
        }

        // 3. Accumulation buffers
        let desc = MTLTextureDescriptor()
        desc.pixelFormat = .rgba32Float
        desc.width = width
        desc.height = height
        desc.usage = [.shaderRead, .shaderWrite]
        guard let sumTex = device.makeTexture(descriptor: desc),
              let sumSqTex = device.makeTexture(descriptor: desc),
              let countTex = device.makeTexture(descriptor: desc) else {
            throw StackError.textureCreationFailed
        }

        guard let accumulateFunc = library.makeFunction(name: "accumulateWithShift") else {
            throw StackError.shaderNotFound("accumulateWithShift")
        }
        let accumulatePSO = try await device.makeComputePipelineState(function: accumulateFunc)

        for (i, tex) in textures.enumerated() {
            guard let cb = commandQueue.makeCommandBuffer(),
                  let enc = cb.makeComputeCommandEncoder() else {
                throw StackError.pipelineCreationFailed
            }
            enc.setComputePipelineState(accumulatePSO)
            enc.setTexture(sumTex, index: 0)
            enc.setTexture(sumSqTex, index: 1)
            enc.setTexture(countTex, index: 2)
            enc.setTexture(tex, index: 3)

            var shift = shifts[i]
            enc.setBytes(&shift, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)

            let tgW = accumulatePSO.threadExecutionWidth
            let tgH = max(1, accumulatePSO.maxTotalThreadsPerThreadgroup / tgW)
            enc.dispatchThreads(
                MTLSize(width: width, height: height, depth: 1),
                threadsPerThreadgroup: MTLSize(width: tgW, height: tgH, depth: 1)
            )
            enc.endEncoding()
            cb.commit()
            await cb.waitUntilCompleted()
            onProgress(0.45 + Double(i + 1) / Double(textures.count) * 0.35)
        }

        // 4. Output pass
        guard let sigmaFunc = library.makeFunction(name: "sigmaClipOutput") else {
            throw StackError.shaderNotFound("sigmaClipOutput")
        }
        let sigmaPSO = try await device.makeComputePipelineState(function: sigmaFunc)

        let outDesc = MTLTextureDescriptor()
        outDesc.pixelFormat = .rgba8Unorm
        outDesc.width = width
        outDesc.height = height
        outDesc.usage = [.shaderRead, .shaderWrite]
        guard let outTex = device.makeTexture(descriptor: outDesc) else {
            throw StackError.textureCreationFailed
        }

        guard let cb2 = commandQueue.makeCommandBuffer(),
              let enc2 = cb2.makeComputeCommandEncoder() else {
            throw StackError.pipelineCreationFailed
        }

        var sigmaVal = sigma
        enc2.setComputePipelineState(sigmaPSO)
        enc2.setTexture(sumTex, index: 0)
        enc2.setTexture(sumSqTex, index: 1)
        enc2.setTexture(countTex, index: 2)
        enc2.setTexture(outTex, index: 3)
        enc2.setBytes(&sigmaVal, length: MemoryLayout<Float>.stride, index: 0)

        let tgW2 = sigmaPSO.threadExecutionWidth
        let tgH2 = max(1, sigmaPSO.maxTotalThreadsPerThreadgroup / tgW2)
        enc2.dispatchThreads(
            MTLSize(width: width, height: height, depth: 1),
            threadsPerThreadgroup: MTLSize(width: tgW2, height: tgH2, depth: 1)
        )
        enc2.endEncoding()

        await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            cb2.addCompletedHandler { _ in cont.resume() }
            cb2.commit()
        }
        onProgress(0.9)

        return try cgImage(from: outTex)
    }

    // MARK: - Alignment (downscaled brute-force SSD on GPU, parallel threads)

    private func computeShifts(reference: MTLTexture, others: [MTLTexture]) async throws -> [SIMD2<Float>] {
        let smallW = max(1, reference.width / alignScale)
        let smallH = max(1, reference.height / alignScale)

        let smallRef = try downscaleTexture(reference, toWidth: smallW, height: smallH)
        var smallOthers: [MTLTexture] = []
        for other in others {
            smallOthers.append(try downscaleTexture(other, toWidth: smallW, height: smallH))
        }

        guard let ssdFunc = library.makeFunction(name: "computeSSD") else {
            throw StackError.shaderNotFound("computeSSD")
        }
        let ssdPSO = try await device.makeComputePipelineState(function: ssdFunc)

        // Result texture for per-pixel SSD (same size as downscaled images)
        let resultDesc = MTLTextureDescriptor()
        resultDesc.pixelFormat = .r32Float
        resultDesc.width = smallW
        resultDesc.height = smallH
        resultDesc.usage = [.shaderRead, .shaderWrite]

        var shifts: [SIMD2<Float>] = [.zero]
        let searchRange = 2  // ±2 px on downscaled = ±16 px on full res

        for otherIdx in 0..<smallOthers.count {
            let smallOther = smallOthers[otherIdx]
            var bestShift = SIMD2<Float>(0, 0)
            var bestSSD: Float = .greatestFiniteMagnitude

            for dy in -searchRange...searchRange {
                for dx in -searchRange...searchRange {
                    guard let cb = commandQueue.makeCommandBuffer(),
                          let enc = cb.makeComputeCommandEncoder(),
                          let resultTex = device.makeTexture(descriptor: resultDesc) else {
                        throw StackError.pipelineCreationFailed
                    }

                    var shift = SIMD2<Float>(Float(dx), Float(dy))
                    enc.setComputePipelineState(ssdPSO)
                    enc.setTexture(smallRef, index: 0)
                    enc.setTexture(smallOther, index: 1)
                    enc.setTexture(resultTex, index: 2)
                    enc.setBytes(&shift, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)

                    let tgW = ssdPSO.threadExecutionWidth
                    let tgH = max(1, ssdPSO.maxTotalThreadsPerThreadgroup / tgW)
                    enc.dispatchThreads(
                        MTLSize(width: smallW, height: smallH, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: tgW, height: tgH, depth: 1)
                    )
                    enc.endEncoding()
                    cb.commit()
                    await cb.waitUntilCompleted()

                    // Read per-pixel SSD and average
                    let ssdBytesPerRow = smallW * 4  // r32Float = 4 bytes per pixel
                    let ssdData = UnsafeMutableRawPointer.allocate(
                        byteCount: ssdBytesPerRow * smallH, alignment: 16
                    )
                    defer { ssdData.deallocate() }
                    resultTex.getBytes(
                        ssdData, bytesPerRow: ssdBytesPerRow,
                        from: MTLRegionMake2D(0, 0, smallW, smallH),
                        mipmapLevel: 0
                    )

                    var sumSSD: Float = 0
                    var count: Float = 0
                    let ptr = ssdData.bindMemory(to: Float.self, capacity: smallW * smallH)
                    for y in 0..<smallH {
                        for x in 0..<smallW {
                            let v = ptr[y * smallW + x]
                            if v > 0 {
                                sumSSD += v
                                count += 1
                            }
                        }
                    }
                    let avgSSD = count > 0 ? sumSSD / count : .greatestFiniteMagnitude

                    if avgSSD < bestSSD {
                        bestSSD = avgSSD
                        bestShift = SIMD2<Float>(Float(dx) * Float(alignScale),
                                                  Float(dy) * Float(alignScale))
                    }
                }
            }
            shifts.append(bestShift)
        }
        return shifts
    }

    private func downscaleTexture(_ tex: MTLTexture, toWidth: Int, height: Int) throws -> MTLTexture {
        let fullBytesPerRow = tex.width * 4
        let fullData = UnsafeMutableRawPointer.allocate(byteCount: fullBytesPerRow * tex.height, alignment: 16)
        defer { fullData.deallocate() }
        tex.getBytes(fullData, bytesPerRow: fullBytesPerRow,
                     from: MTLRegionMake2D(0, 0, tex.width, tex.height),
                     mipmapLevel: 0)

        guard let fullCI = CGContext(
            data: fullData, width: tex.width, height: tex.height,
            bitsPerComponent: 8, bytesPerRow: fullBytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let fullImg = fullCI.makeImage() else {
            throw StackError.textureCreationFailed
        }

        let smallBytesPerRow = toWidth * 4
        let smallData = UnsafeMutableRawPointer.allocate(byteCount: smallBytesPerRow * height, alignment: 16)
        defer { smallData.deallocate() }

        guard let smallCI = CGContext(
            data: smallData, width: toWidth, height: height,
            bitsPerComponent: 8, bytesPerRow: smallBytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw StackError.textureCreationFailed
        }
        smallCI.interpolationQuality = .medium
        smallCI.draw(fullImg, in: CGRect(x: 0, y: 0, width: toWidth, height: height))

        let desc = MTLTextureDescriptor()
        desc.pixelFormat = .rgba8Unorm
        desc.width = toWidth
        desc.height = height
        desc.usage = [.shaderRead, .shaderWrite]
        guard let smallTex = device.makeTexture(descriptor: desc) else {
            throw StackError.textureCreationFailed
        }
        smallTex.replace(
            region: MTLRegionMake2D(0, 0, toWidth, height),
            mipmapLevel: 0,
            withBytes: smallData,
            bytesPerRow: smallBytesPerRow
        )
        return smallTex
    }

    // MARK: - Texture helpers

    private func makeTexture(from cg: CGImage) throws -> MTLTexture {
        let width = cg.width
        let height = cg.height
        let bytesPerRow = width * 4

        let data = UnsafeMutableRawPointer.allocate(byteCount: bytesPerRow * height, alignment: 16)
        defer { data.deallocate() }

        guard let ci = CGContext(
            data: data, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw StackError.textureCreationFailed
        }
        ci.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))

        let desc = MTLTextureDescriptor()
        desc.pixelFormat = .rgba8Unorm
        desc.width = width
        desc.height = height
        desc.usage = [.shaderRead, .shaderWrite]
        guard let tex = device.makeTexture(descriptor: desc) else {
            throw StackError.textureCreationFailed
        }
        tex.replace(
            region: MTLRegionMake2D(0, 0, width, height),
            mipmapLevel: 0,
            withBytes: data,
            bytesPerRow: bytesPerRow
        )
        return tex
    }

    private func cgImage(from tex: MTLTexture) throws -> CGImage {
        let width = tex.width
        let height = tex.height
        let bytesPerRow = width * 4

        let data = UnsafeMutableRawPointer.allocate(byteCount: bytesPerRow * height, alignment: 16)
        defer { data.deallocate() }

        tex.getBytes(data, bytesPerRow: bytesPerRow,
                     from: MTLRegionMake2D(0, 0, width, height),
                     mipmapLevel: 0)

        // Create a self-owning context (data:nil) and draw into it to guarantee copy
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        guard let srcContext = CGContext(
            data: data, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ), let srcImg = srcContext.makeImage() else {
            throw StackError.textureCreationFailed
        }

        guard let dstContext = CGContext(
            data: nil, width: width, height: height,
            bitsPerComponent: 8, bytesPerRow: 0,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw StackError.textureCreationFailed
        }
        dstContext.draw(srcImg, in: CGRect(x: 0, y: 0, width: width, height: height))

        guard let result = dstContext.makeImage() else {
            throw StackError.textureCreationFailed
        }
        return result
    }
}

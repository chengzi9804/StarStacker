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

        // Validate all images have matching dimensions
        for (i, img) in images.enumerated() where img.width != width || img.height != height {
            throw StackError.imageSizeMismatch
        }

        // Convert all CGImages to Metal textures
        var textures: [MTLTexture] = []
        for (i, cg) in images.enumerated() {
            let tex = try makeTexture(from: cg)
            textures.append(tex)
            onProgress(Double(i + 1) / Double(images.count) * 0.25)
        }

        // Compute alignment shifts (brute-force in Metal for speed)
        var shifts: [SIMD2<Float>] = .init(repeating: .zero, count: textures.count)
        if align && textures.count > 1 {
            shifts = try await computeShiftsGPU(reference: textures[0], others: Array(textures.dropFirst()))
            onProgress(0.55)
        }

        // Accumulation buffers (float for precision)
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
        let accumulatePSO = try device.makeComputePipelineState(function: accumulateFunc)

        // Each frame gets its own command buffer + encoder to avoid binding issues
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
            onProgress(0.55 + Double(i + 1) / Double(textures.count) * 0.25)
        }

        // Sigma-clip output pass
        guard let sigmaFunc = library.makeFunction(name: "sigmaClipOutput") else {
            throw StackError.shaderNotFound("sigmaClipOutput")
        }
        let sigmaPSO = try device.makeComputePipelineState(function: sigmaFunc)

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

        return try cgImage(from: outTex, width: width, height: height)
    }

    // MARK: - Alignment via GPU (brute-force SSD in compute shader)

    private func computeShiftsGPU(reference: MTLTexture, others: [MTLTexture]) async throws -> [SIMD2<Float>] {
        guard let ssdFunc = library.makeFunction(name: "computeSSD") else {
            throw StackError.shaderNotFound("computeSSD")
        }
        let ssdPSO = try device.makeComputePipelineState(function: ssdFunc)

        var shifts: [SIMD2<Float>] = [.zero]
        let range: Float = 8.0

        for other in others {
            var bestShift = SIMD2<Float>(0, 0)
            var bestSSD: Float = .greatestFiniteMagnitude

            // Search integer shifts in range [-8, 8]
            for dy in -8...8 {
                for dx in -8...8 {
                    guard let cb = commandQueue.makeCommandBuffer(),
                          let enc = cb.makeComputeCommandEncoder() else {
                        throw StackError.pipelineCreationFailed
                    }

                    // 1x1 buffer to receive SSD result
                    let resultDesc = MTLTextureDescriptor()
                    resultDesc.pixelFormat = .rgba32Float
                    resultDesc.width = 1
                    resultDesc.height = 1
                    resultDesc.usage = [.shaderRead, .shaderWrite]
                    guard let resultTex = device.makeTexture(descriptor: resultDesc) else {
                        throw StackError.textureCreationFailed
                    }

                    var shift = SIMD2<Float>(Float(dx), Float(dy))
                    enc.setComputePipelineState(ssdPSO)
                    enc.setTexture(reference, index: 0)
                    enc.setTexture(other, index: 1)
                    enc.setTexture(resultTex, index: 2)
                    enc.setBytes(&shift, length: MemoryLayout<SIMD2<Float>>.stride, index: 0)
                    enc.dispatchThreads(
                        MTLSize(width: 1, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
                    )
                    enc.endEncoding()
                    cb.commit()
                    await cb.waitUntilCompleted()

                    // Read 1x1 result
                    var result: Float = 0
                    resultTex.getBytes(
                        &result,
                        bytesPerRow: 16,
                        from: MTLRegionMake2D(CGRect(x: 0, y: 0, width: 1, height: 1), 0),
                        mipmapLevel: 0
                    )

                    if result < bestSSD {
                        bestSSD = result
                        bestShift = SIMD2<Float>(Float(dx), Float(dy))
                    }
                }
            }
            shifts.append(bestShift)
        }
        return shifts
    }

    // MARK: - Texture helpers

    private func makeTexture(from cg: CGImage) throws -> MTLTexture {
        let width = cg.width
        let height = cg.height
        let bytesPerRow = width * 4

        // Allocate our own buffer so we control lifetime
        let data = UnsafeMutableRawPointer.allocate(byteCount: bytesPerRow * height, alignment: 16)
        defer { data.deallocate() }

        guard let ci = CGContext(
            data: data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
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
            region: MTLRegionMake2D(CGRect(x: 0, y: 0, width: width, height: height), 0),
            mipmapLevel: 0,
            withBytes: data,
            bytesPerRow: bytesPerRow
        )
        return tex
    }

    private func cgImage(from tex: MTLTexture, width: Int, height: Int) throws -> CGImage {
        let bytesPerRow = width * 4
        let totalBytes = bytesPerRow * height

        // Allocate buffer that outlives the function; CGContext will copy it into CGImage
        let data = UnsafeMutableRawPointer.allocate(byteCount: totalBytes, alignment: 16)
        defer { data.deallocate() }

        let region = MTLRegionMake2D(CGRect(x: 0, y: 0, width: width, height: height), 0)
        tex.getBytes(data, bytesPerRow: bytesPerRow, from: region, mipmapLevel: 0)

        guard let ci = CGContext(
            data: data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            throw StackError.textureCreationFailed
        }

        // CGContext.makeImage() copies the bitmap data into a new CGImage,
        // so the deferred deallocation of `data` after this return is safe.
        guard let image = ci.makeImage() else {
            throw StackError.textureCreationFailed
        }
        return image
    }
}

import CoreGraphics
import ImageIO
import UniformTypeIdentifiers

public enum DNGLoader {
    /// Load a DNG/RAW file from URL and return a CGImage
    public static func load(url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else {
            return nil
        }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    /// Load image from Data (e.g. from PhotosPicker)
    public static func load(data: Data) -> CGImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            return nil
        }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}

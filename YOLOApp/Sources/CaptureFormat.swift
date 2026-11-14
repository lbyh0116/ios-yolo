import AVFoundation
import CoreMedia

/// One capture configuration the device can actually deliver: resolution class × frame rate.
///
/// `format` is the concrete `AVCaptureDevice.Format` the option was derived from, so applying an
/// option never has to re-search the device and can never pick something the device rejected.
struct CaptureFormatOption: Identifiable {
    let id: String
    let title: String
    let fps: Double
    let width: Int
    let height: Int
    let format: AVCaptureDevice.Format

    var subtitle: String { "\(Int(fps)) fps" }

    var menuTitle: String { "\(title) · \(Int(fps))fps" }

    var pixelCount: Int { width * height }
}

extension CaptureFormatOption {
    /// Options for the resolutions a phone or iPad camera commonly offers, filtered to the ones that
    /// can deliver 420f (full-range NV12) — the pixel format AprilTag reads its luma plane from.
    static func enumerate(_ device: AVCaptureDevice) -> [CaptureFormatOption] {
        let classes: [(width: Int, height: Int, title: String)] = [
            (3840, 2160, "4K"),
            (1920, 1080, "1080p"),
            (1280, 720, "720p"),
            (640, 480, "480p"),
        ]
        var options: [CaptureFormatOption] = []
        for resolution in classes {
            let candidates = device.formats.filter { format in
                let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
                return Int(dimensions.width) == resolution.width
                    && Int(dimensions.height) == resolution.height
                    && format.supportsFullRange420
            }
            // Several formats can share a resolution; the fastest one wins and its rate range
            // decides which frame rates are offered below.
            guard let format = candidates.max(by: { $0.maxFrameRate < $1.maxFrameRate }) else { continue }
            for fps in [24.0, 30.0, 60.0] where fps <= format.maxFrameRate + 0.01 && fps >= format.minFrameRate - 0.01 {
                options.append(CaptureFormatOption(
                    id: "\(resolution.width)x\(resolution.height)@\(Int(fps))",
                    title: resolution.title,
                    fps: fps,
                    width: resolution.width,
                    height: resolution.height,
                    format: format
                ))
            }
        }
        return options.sorted { lhs, rhs in
            lhs.pixelCount == rhs.pixelCount ? lhs.fps > rhs.fps : lhs.pixelCount > rhs.pixelCount
        }
    }
}

struct CaptureFormatGroup: Identifiable {
    let title: String
    let options: [CaptureFormatOption]

    var id: String { title }

    /// Consecutive options of the same resolution class, as `enumerate()` orders them.
    static func groups(of options: [CaptureFormatOption]) -> [CaptureFormatGroup] {
        var groups: [CaptureFormatGroup] = []
        for option in options {
            if let last = groups.last, last.title == option.title {
                groups[groups.count - 1] = CaptureFormatGroup(title: last.title, options: last.options + [option])
            } else {
                groups.append(CaptureFormatGroup(title: option.title, options: [option]))
            }
        }
        return groups
    }
}

extension AVCaptureDevice.Format {
    var maxFrameRate: Double { videoSupportedFrameRateRanges.map(\.maxFrameRate).max() ?? 0 }
    var minFrameRate: Double { videoSupportedFrameRateRanges.map(\.minFrameRate).min() ?? 0 }

    /// The pixel format AprilTag reads its luma plane from must survive whatever resolution we pick.
    var supportsFullRange420: Bool {
        supportedPixelFormatTypes.contains { ($0 as NSNumber).uint32Value == kCVPixelFormatType_420YpCbCr8BiPlanarFullRange }
    }
}

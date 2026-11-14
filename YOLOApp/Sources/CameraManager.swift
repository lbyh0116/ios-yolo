import AVFoundation
import CoreVideo
import simd

/// Pinhole intrinsics in pixels of the delivered (rotated) frame.
struct CameraIntrinsics {
    var fx: Double
    var fy: Double
    var cx: Double
    var cy: Double
    /// True when taken from AVFoundation's per-frame matrix, false when estimated from the field of view.
    var isFromCamera: Bool
}

struct CameraFrame {
    let pixelBuffer: CVPixelBuffer
    let intrinsics: CameraIntrinsics
    /// Presentation time, seconds.
    let timestamp: Double
}

/// Back-camera capture delivering 420f (full-range NV12) frames on `videoQueue`, rotated to match the
/// preview so detection always runs on an upright frame whatever the interface orientation is.
final class CameraManager: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let session = AVCaptureSession()
    let videoQueue = DispatchQueue(label: "yolo.camera.video", qos: .userInitiated)
    /// Called on `videoQueue` for every frame that isn't dropped.
    var onFrame: ((CameraFrame) -> Void)?

    private let sessionQueue = DispatchQueue(label: "yolo.camera.session")
    private let output = AVCaptureVideoDataOutput()
    private let rotationLock = NSLock()
    private var isConfigured = false
    /// Horizontal field of view of the active format in degrees (along the sensor's long side).
    private var fieldOfView: Float = 70
    /// Right-angle rotation applied to the delivered frames, kept in sync with the preview layer.
    private var rotationAngle: CGFloat = 90

    /// Written on `sessionQueue` during `configure()`; read afterwards on the main queue.
    private(set) var device: AVCaptureDevice?
    private(set) var formats: [CaptureFormatOption] = []
    private(set) var activeFormatID: String?

    private static let savedFormatKey = "captureFormatID"

    /// Requests permission, configures the session and starts it. `completion` runs on the main queue.
    func start(completion: @escaping (String?) -> Void) {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            guard granted else {
                DispatchQueue.main.async { completion("未获得相机权限，请在 设置 > 隐私 > 相机 中开启") }
                return
            }
            self.sessionQueue.async {
                if !self.isConfigured {
                    if let error = self.configure() {
                        DispatchQueue.main.async { completion(error) }
                        return
                    }
                    self.isConfigured = true
                }
                if !self.session.isRunning { self.session.startRunning() }
                DispatchQueue.main.async { completion(nil) }
            }
        }
    }

    func stop() {
        sessionQueue.async {
            if self.session.isRunning { self.session.stopRunning() }
        }
    }

    /// Switches to another resolution / frame rate; the preview keeps running through the change.
    func selectFormat(_ option: CaptureFormatOption) {
        sessionQueue.async {
            guard let device = self.device else { return }
            self.session.beginConfiguration()
            defer { self.session.commitConfiguration() }
            self.apply(option, to: device)
        }
    }

    /// Rotates the delivered frames; the preview layer is given the same value so boxes stay aligned.
    func setRotationAngle(_ degrees: CGFloat) {
        let snapped = (degrees / 90).rounded() * 90
        rotationLock.lock()
        rotationAngle = snapped
        rotationLock.unlock()
        if let connection = output.connection(with: .video), connection.isVideoRotationAngleSupported(snapped) {
            connection.videoRotationAngle = snapped
        }
    }

    private func configure() -> String? {
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return "找不到后置摄像头"
        }
        self.device = device
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        // `.inputPriority` is required to pick an arbitrary `activeFormat`: 4K is not a session preset.
        if session.canSetSessionPreset(.inputPriority) {
            session.sessionPreset = .inputPriority
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            guard session.canAddInput(input) else { return "无法添加相机输入" }
            session.addInput(input)
        } catch {
            return "相机打开失败：\(error.localizedDescription)"
        }

        // Luma plane feeds AprilTag directly; Vision converts for YOLO.
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: videoQueue)
        guard session.canAddOutput(output) else { return "无法添加视频输出" }
        session.addOutput(output)

        if let connection = output.connection(with: .video) {
            if connection.isVideoRotationAngleSupported(90) {
                connection.videoRotationAngle = 90 // portrait until the preview reports the real angle
            }
            if connection.isCameraIntrinsicMatrixDeliverySupported {
                connection.isCameraIntrinsicMatrixDeliveryEnabled = true
            }
        }

        formats = CaptureFormatOption.enumerate(device)
        if let option = preferredFormat() {
            apply(option, to: device)
        }
        return nil
    }

    private func preferredFormat() -> CaptureFormatOption? {
        if let saved = UserDefaults.standard.string(forKey: Self.savedFormatKey),
           let match = formats.first(where: { $0.id == saved }) {
            return match
        }
        return formats.first { $0.id == "1920x1080@60" } ?? formats.first
    }

    /// Call inside a session configuration block.
    private func apply(_ option: CaptureFormatOption, to device: AVCaptureDevice) {
        let maxFPS = option.format.maxFrameRate
        let minFPS = max(option.format.minFrameRate, 1)
        let fps = min(max(option.fps, minFPS), maxFPS > 0 ? maxFPS : option.fps)
        do {
            try device.lockForConfiguration()
            device.activeFormat = option.format
            let duration = CMTime(value: 1, timescale: CMTimeScale(fps.rounded()))
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration
            device.unlockForConfiguration()
        } catch {
            return // keep the previous format; the UI keeps showing what is really active
        }
        fieldOfView = option.format.videoFieldOfView
        activeFormatID = option.id
        UserDefaults.standard.set(option.id, forKey: Self.savedFormatKey)
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let width = Double(CVPixelBufferGetWidth(pixelBuffer))
        let height = Double(CVPixelBufferGetHeight(pixelBuffer))
        rotationLock.lock()
        let angle = rotationAngle
        rotationLock.unlock()
        let matrix = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil) as? Data
        let intrinsics = matrix.flatMap { parseIntrinsics($0, width: width, height: height, angle: angle) }
            ?? estimatedIntrinsics(width: width, height: height)
        let timestamp = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        onFrame?(CameraFrame(pixelBuffer: pixelBuffer, intrinsics: intrinsics, timestamp: timestamp))
    }

    /// AVFoundation describes intrinsics against the sensor's native landscape frame, so the matrix has
    /// to follow whatever rotation the delivered frames were given.
    private func parseIntrinsics(_ data: Data, width: Double, height: Double, angle: CGFloat) -> CameraIntrinsics? {
        guard data.count >= MemoryLayout<matrix_float3x3>.size else { return nil }
        let m = data.withUnsafeBytes { $0.loadUnaligned(as: matrix_float3x3.self) }
        var fx = Double(m.columns.0.x), fy = Double(m.columns.1.y)
        var cx = Double(m.columns.2.x), cy = Double(m.columns.2.y)
        guard fx > 0, fy > 0 else { return nil }

        let normalized = Int(((angle.truncatingRemainder(dividingBy: 360)) + 360).truncatingRemainder(dividingBy: 360))
        let matrixIsLandscape = cx > cy
        let frameIsLandscape = width > height

        if matrixIsLandscape != frameIsLandscape, normalized == 90 || normalized == 270 {
            if normalized == 90 {
                (fx, fy, cx, cy) = (fy, fx, width - cy, cx)
            } else {
                (fx, fy, cx, cy) = (fy, fx, cy, height - cx)
            }
        } else if matrixIsLandscape == frameIsLandscape, normalized == 180 {
            cx = width - cx
            cy = height - cy
        }

        let plausible = abs(cx - width / 2) < width * 0.2 && abs(cy - height / 2) < height * 0.2
        return plausible ? CameraIntrinsics(fx: fx, fy: fy, cx: cx, cy: cy, isFromCamera: true) : nil
    }

    private func estimatedIntrinsics(width: Double, height: Double) -> CameraIntrinsics {
        let focal = max(width, height) / 2 / tan(Double(fieldOfView) * .pi / 360)
        return CameraIntrinsics(fx: focal, fy: focal, cx: width / 2, cy: height / 2, isFromCamera: false)
    }
}

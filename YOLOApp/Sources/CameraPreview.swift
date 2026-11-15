import AVFoundation
import SwiftUI

/// How the camera image fills the preview. `fill` crops, `fit` letterboxes; the overlay mapping follows
/// whichever is selected so detection boxes stay glued to the image.
enum PreviewFillMode: String, CaseIterable, Identifiable {
    case fill
    case fit

    var id: String { rawValue }

    var title: String { self == .fill ? "填满" : "适配" }

    var gravity: AVLayerVideoGravity { self == .fill ? .resizeAspectFill : .resizeAspect }
}

struct CameraPreview: UIViewRepresentable {
    let session: AVCaptureSession
    /// Back-camera device; used by the rotation coordinator fallback.
    let device: AVCaptureDevice?
    let fillMode: PreviewFillMode
    /// Degrees forced by the user (0/90/180/270); nil means follow the window.
    let forcedAngle: CGFloat?
    /// Reports the rotation actually applied so the data output can match it.
    let onRotationAngle: (CGFloat) -> Void

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.gravity = fillMode.gravity
        view.onRotationAngle = onRotationAngle
        view.forcedAngle = forcedAngle
        view.device = device
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.gravity = fillMode.gravity
        uiView.onRotationAngle = onRotationAngle
        uiView.forcedAngle = forcedAngle
        uiView.device = device
        uiView.scheduleRotationSync()
    }

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

        var onRotationAngle: ((CGFloat) -> Void)?

        /// Re-applied on every layout: a rotated connection can otherwise leave the layer letterboxed
        /// even though "fill" is selected.
        var gravity: AVLayerVideoGravity = .resizeAspectFill {
            didSet { previewLayer.videoGravity = gravity }
        }

        var forcedAngle: CGFloat? {
            didSet { if forcedAngle != oldValue { lastAngle = -1; scheduleRotationSync() } }
        }

        var device: AVCaptureDevice? {
            didSet {
                if device !== oldValue {
                    coordinator = nil
                    lastAngle = -1
                }
            }
        }

        private var coordinator: AVCaptureDevice.RotationCoordinator?
        private var lastAngle: CGFloat = -1
        private var pendingSync: DispatchWorkItem?
        private var orientationObserver: NSObjectProtocol?

        override func didMoveToWindow() {
            super.didMoveToWindow()
            if window != nil {
                startObservingOrientation()
            } else {
                stopObservingOrientation()
            }
            scheduleRotationSync()
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            previewLayer.videoGravity = gravity
            scheduleRotationSync()
        }

        private func startObservingOrientation() {
            guard orientationObserver == nil else { return }
            UIDevice.current.beginGeneratingDeviceOrientationNotifications()
            orientationObserver = NotificationCenter.default.addObserver(
                forName: UIDevice.orientationDidChangeNotification, object: nil, queue: .main
            ) { [weak self] _ in
                self?.lastAngle = -1
                self?.scheduleRotationSync()
            }
        }

        private func stopObservingOrientation() {
            if let orientationObserver {
                NotificationCenter.default.removeObserver(orientationObserver)
                self.orientationObserver = nil
            }
        }

        deinit {
            if let orientationObserver {
                NotificationCenter.default.removeObserver(orientationObserver)
            }
        }

        /// Resizing fires `layoutSubviews` dozens of times per second, and every rotation change
        /// restarts the capture pipeline — which is exactly the hiccup you see when a window grows or
        /// shrinks. So the angle is applied only after the geometry has been stable for 0.15 s.
        func scheduleRotationSync() {
            pendingSync?.cancel()
            let work = DispatchWorkItem { [weak self] in self?.syncRotation() }
            pendingSync = work
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15, execute: work)
        }

        /// Keeps the preview and the delivered frames at the same right-angle rotation.
        ///
        /// The window's own interface orientation is the primary signal: if the UI rotated, that value
        /// is right, whereas `RotationCoordinator` reads the host device and gets confused inside
        /// containers such as LiveContainer. The coordinator is only a fallback for when the scene is
        /// not available yet, and a user-forced angle wins over both.
        func syncRotation() {
            guard window != nil else { return }
            let snapped = (desiredAngle() / 90).rounded() * 90
            guard snapped != lastAngle else { return }
            lastAngle = snapped
            if let connection = previewLayer.connection, connection.isVideoRotationAngleSupported(snapped) {
                connection.videoRotationAngle = snapped
                previewLayer.videoGravity = gravity
            }
            onRotationAngle?(snapped)
        }

        private func desiredAngle() -> CGFloat {
            if let forcedAngle { return forcedAngle }
            if let scene = window?.windowScene {
                if #available(iOS 16.0, *) {
                    return Self.angle(for: scene.effectiveGeometry.interfaceOrientation)
                }
                return Self.angle(for: scene.interfaceOrientation)
            }
            if coordinator == nil, let device {
                coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
            }
            return coordinator?.videoRotationAngleForHorizonLevelPreview ?? 90
        }

        /// Apple's back-camera mapping (as used by AVCam): portrait buffers at 90°.
        static func angle(for orientation: UIInterfaceOrientation) -> CGFloat {
            switch orientation {
            case .portrait: return 90
            case .portraitUpsideDown: return 270
            case .landscapeLeft: return 180
            case .landscapeRight: return 0
            default: return 90
            }
        }
    }
}

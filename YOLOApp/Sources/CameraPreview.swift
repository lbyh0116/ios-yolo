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
    /// Back-camera device; needed to build the rotation coordinator once the session is configured.
    let device: AVCaptureDevice?
    let fillMode: PreviewFillMode
    /// Reports the right-angle rotation the preview uses so the data output can match it.
    let onRotationAngle: (CGFloat) -> Void

    func makeUIView(context: Context) -> PreviewView {
        let view = PreviewView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = fillMode.gravity
        view.onRotationAngle = onRotationAngle
        view.device = device
        return view
    }

    func updateUIView(_ uiView: PreviewView, context: Context) {
        uiView.previewLayer.videoGravity = fillMode.gravity
        uiView.onRotationAngle = onRotationAngle
        uiView.device = device
        uiView.syncRotation()
    }

    final class PreviewView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }

        var onRotationAngle: ((CGFloat) -> Void)?

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

        override func didMoveToWindow() {
            super.didMoveToWindow()
            syncRotation()
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            syncRotation()
        }

        /// Keeps the preview and the delivered frames at the same right-angle rotation.
        ///
        /// `RotationCoordinator` derives the angle from the device and the preview's geometry, which is
        /// what makes this work in a resizable iPad window as well as on a rotating phone. The angle is
        /// snapped to a multiple of 90° so the intrinsic matrix can be rotated exactly.
        func syncRotation() {
            guard let device, window != nil else { return }
            if coordinator == nil {
                coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: previewLayer)
            }
            guard let coordinator else { return }
            let snapped = (coordinator.videoRotationAngleForHorizonLevelPreview / 90).rounded() * 90
            guard snapped != lastAngle else { return }
            lastAngle = snapped
            if let connection = previewLayer.connection, connection.isVideoRotationAngleSupported(snapped) {
                connection.videoRotationAngle = snapped
            }
            onRotationAngle?(snapped)
        }
    }
}

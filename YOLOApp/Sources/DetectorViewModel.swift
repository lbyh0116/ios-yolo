import AVFoundation
import Combine
import CoreGraphics
import CoreVideo
import Foundation
import QuartzCore

struct FrameStats {
    var fps: Double = 0
    var inferenceMs: Double = 0
    var postprocessMs: Double = 0
    var objectCount = 0
    var tagMs: Double = 0
    var tagCount = 0
    var intrinsicsFromCamera = false
}

enum DetectionMode: String, CaseIterable, Identifiable {
    case yolo
    case aprilTag
    case both

    var id: String { rawValue }

    var title: String {
        switch self {
        case .yolo: return "YOLO"
        case .aprilTag: return "AprilTag"
        case .both: return "同时"
        }
    }

    var runsYOLO: Bool { self != .aprilTag }
    var runsAprilTag: Bool { self != .yolo }
}

struct BenchmarkResult: Identifiable {
    let mode: ComputeMode
    var loadMs: Double?
    var stats: BenchmarkStats?
    var breakdown: DeviceBreakdown?
    var error: String?

    var id: String { mode.id }
}

final class DetectorViewModel: ObservableObject {
    let models = ModelInfo.bundled()
    let camera = CameraManager()

    @Published var selectedModel: ModelInfo? {
        didSet { if selectedModel != oldValue { loadModel() } }
    }
    @Published var computeMode: ComputeMode = .neuralEngine {
        didSet { if computeMode != oldValue { loadModel() } }
    }
    @Published var confidence: Double = 0.25 {
        didSet {
            let value = Float(confidence)
            camera.videoQueue.async { self.confidenceThreshold = value }
        }
    }

    @Published var detectionMode: DetectionMode = .yolo {
        didSet {
            let mode = detectionMode
            camera.videoQueue.async {
                self.activeMode = mode
                if !mode.runsAprilTag { self.tagTracker.reset() }
            }
        }
    }
    /// Edge length of the tag's outer black square, in centimeters.
    @Published var tagSizeCM: Double = 6.9 {
        didSet {
            let meters = tagSizeCM / 100
            camera.videoQueue.async { self.activeTagSize = meters }
        }
    }
    /// AprilTag pose smoothing, 0 = most responsive, 1 = steadiest.
    @Published var tagSmoothing: Double = 0.5 {
        didSet {
            let value = tagSmoothing
            camera.videoQueue.async { self.tagTracker.smoothing = value }
        }
    }
    @Published var tagOrientation: TagPatternOrientation = .rotated180 {
        didSet {
            let orientation = tagOrientation
            camera.videoQueue.async {
                self.activeTagOrientation = orientation
                self.tagTracker.reset()
            }
        }
    }

    /// Stick-figure mode: switches to a pose model so the overlay can draw skeletons.
    @Published var stickFigure = false {
        didSet { if stickFigure != oldValue { applyStickFigure() } }
    }

    @Published private(set) var detections: [Detection] = []
    @Published private(set) var tags: [TagOverlay] = []
    @Published private(set) var imageSize = CGSize(width: 1080, height: 1920)
    @Published private(set) var stats = FrameStats()
    @Published private(set) var isLoading = false
    @Published private(set) var statusMessage: String?
    @Published private(set) var errorMessage: String?

    @Published private(set) var benchmarkResults: [BenchmarkResult] = []
    @Published private(set) var isBenchmarking = false

    /// Back camera, published once the session has been configured (drives the rotation coordinator).
    @Published private(set) var device: AVCaptureDevice?
    /// Capture formats the current device can deliver, ordered 4K → 480p, fastest frame rate first.
    @Published private(set) var formats: [CaptureFormatOption] = []
    @Published private(set) var activeFormatID: String?

    var activeFormat: CaptureFormatOption? { formats.first { $0.id == activeFormatID } }

    var formatLabel: String {
        guard let format = activeFormat else { return "格式" }
        return "\(format.title) · \(Int(format.fps))fps"
    }

    /// Switches resolution / frame rate; the displayed label follows the camera's real state.
    func selectFormat(_ option: CaptureFormatOption) {
        activeFormatID = option.id
        camera.selectFormat(option)
    }

    /// Called by the preview layer with the right-angle rotation it settled on.
    func updateRotationAngle(_ degrees: CGFloat) {
        camera.setRotationAngle(degrees)
    }

    /// User-forced rotation: -1 = follow the window, otherwise a locked 0/90/180/270.
    @Published var rotationOverride: Int = UserDefaults.standard.object(forKey: DetectorViewModel.rotationKey) as? Int ?? -1 {
        didSet {
            guard rotationOverride != oldValue else { return }
            UserDefaults.standard.set(rotationOverride, forKey: DetectorViewModel.rotationKey)
            if rotationOverride >= 0 {
                camera.setRotationAngle(CGFloat(rotationOverride))
            }
        }
    }

    var forcedRotationAngle: CGFloat? { rotationOverride >= 0 ? CGFloat(rotationOverride) : nil }

    private static let rotationKey = "rotationOverrideDegrees"

    // MARK: - Model list / stick figure

    /// The picker never mixes flavours: detection models when the skeleton toggle is off, pose models
    /// when it is on.
    var visibleModels: [ModelInfo] { models.filter { $0.isPose == stickFigure } }

    var modelGroups: [ModelGroup] { ModelGroup.groups(of: visibleModels) }

    /// Turns stick-figure mode on/off. Pose models are the only ones that emit keypoints, so switching
    /// the model is part of the toggle rather than a separate pick.
    private func applyStickFigure() {
        if stickFigure {
            guard selectedModel?.isPose != true else { return }
            if let pose = poseModel(matching: selectedModel) ?? models.first(where: { $0.isPose }) {
                selectedModel = pose
            } else {
                stickFigure = false
                errorMessage = "这个包里没有 pose 模型：构建时在 models 里加 yolo26n-pose 之类"
            }
            return
        }
        // Switching off must not leave a pose model selected now that pose entries vanish from the list.
        guard let current = selectedModel, current.isPose else { return }
        selectedModel = models.first { $0.name == current.baseName } ?? models.first { !$0.isPose }
    }

    /// Same scale when bundled ("yolo26m" → "yolo26m-pose"), otherwise the fastest pose model available.
    private func poseModel(matching current: ModelInfo?) -> ModelInfo? {
        guard let current else { return nil }
        if let exact = models.first(where: { $0.name == current.baseName + "-pose" }) { return exact }
        guard let scale = current.scale else { return nil }
        return models.first { $0.isPose && $0.scale == scale }
    }

    private let loadQueue = DispatchQueue(label: "yolo.model.load", qos: .userInitiated)
    private var loadGeneration = 0
    private var hasStarted = false

    // Confined to camera.videoQueue.
    private var detector: YOLODetector?
    private var confidenceThreshold: Float = 0.25
    private var isPaused = false
    private var activeMode: DetectionMode = .yolo
    private var activeTagSize = 0.069
    private var activeTagOrientation: TagPatternOrientation = .rotated180
    private lazy var tagDetector = AprilTagDetector()
    private let tagTracker = AprilTagTracker()
    private var lastFrameTime: CFTimeInterval = 0
    private var smoothedFPS: Double = 0

    init() {
        selectedModel = models.first
        camera.onFrame = { [weak self] frame in self?.process(frame) }
    }

    func start() {
        camera.start { [weak self] error in
            guard let self else { return }
            if let error {
                self.errorMessage = error
                return
            }
            self.device = self.camera.device
            self.formats = self.camera.formats
            self.activeFormatID = self.camera.activeFormatID
        }
        if !hasStarted {
            hasStarted = true
            loadModel()
        }
    }

    // MARK: - Model loading

    private func loadModel() {
        guard let model = selectedModel else {
            errorMessage = "App 内没有模型，请检查构建流程中的模型导出步骤"
            return
        }
        let mode = computeMode
        loadGeneration += 1
        let generation = loadGeneration
        isLoading = true
        statusMessage = nil

        loadQueue.async {
            let result = Result { try YOLODetector(info: model, computeMode: mode) }
            if case let .success(detector) = result {
                self.camera.videoQueue.async {
                    self.detector = detector
                    self.smoothedFPS = 0
                }
            }
            DispatchQueue.main.async {
                guard generation == self.loadGeneration else { return }
                self.isLoading = false
                switch result {
                case let .success(detector):
                    self.errorMessage = nil
                    self.statusMessage = String(format: "%@ 已加载到 %@，用时 %.0f ms", model.name, mode.title, detector.loadTime * 1000)
                case let .failure(error):
                    self.errorMessage = "模型加载失败：\(error.localizedDescription)"
                }
            }
        }
    }

    // MARK: - Per-frame processing (camera.videoQueue)

    private func process(_ frame: CameraFrame) {
        guard !isPaused else { return }
        let pixelBuffer = frame.pixelBuffer
        let size = CGSize(width: CVPixelBufferGetWidth(pixelBuffer), height: CVPixelBufferGetHeight(pixelBuffer))
        var stats = FrameStats()
        var detections: [Detection] = []
        var tags: [TagOverlay] = []
        var failure: String?

        if activeMode.runsYOLO, let detector {
            do {
                let result = try detector.detect(pixelBuffer: pixelBuffer, confidence: confidenceThreshold)
                detections = result.detections
                stats.inferenceMs = result.inferenceMs
                stats.postprocessMs = result.postprocessMs
                stats.objectCount = result.detections.count
            } catch {
                failure = "推理失败：\(error.localizedDescription)"
            }
        }

        if activeMode.runsAprilTag {
            let start = CACurrentMediaTime()
            let found = tagDetector.detect(pixelBuffer: pixelBuffer, intrinsics: frame.intrinsics, tagSize: activeTagSize)
            tags = tagTracker.update(
                with: found,
                timestamp: frame.timestamp.isFinite ? frame.timestamp : start,
                intrinsics: frame.intrinsics,
                imageSize: size,
                tagSize: activeTagSize,
                orientation: activeTagOrientation
            )
            stats.tagMs = (CACurrentMediaTime() - start) * 1000
            stats.tagCount = found.count
            stats.intrinsicsFromCamera = frame.intrinsics.isFromCamera
        }

        let now = CACurrentMediaTime()
        if lastFrameTime > 0 {
            let instant = 1 / max(now - lastFrameTime, 1e-6)
            smoothedFPS = smoothedFPS == 0 ? instant : smoothedFPS * 0.9 + instant * 0.1
        }
        lastFrameTime = now
        stats.fps = smoothedFPS

        DispatchQueue.main.async {
            self.detections = detections
            self.tags = tags
            self.imageSize = size
            self.stats = stats
            if let failure { self.errorMessage = failure }
        }
    }

    // MARK: - Benchmark

    /// Loads the selected model on NPU, GPU and CPU in turn and times raw predictions.
    func runBenchmark(iterations: Int = 50) {
        guard let model = selectedModel, !isBenchmarking else { return }
        isBenchmarking = true
        benchmarkResults = []
        camera.videoQueue.async {
            self.isPaused = true
            self.tagTracker.reset()
            DispatchQueue.main.async {
                self.detections = []
                self.tags = []
            }
        }

        Task.detached(priority: .userInitiated) {
            for mode in [ComputeMode.neuralEngine, .gpu, .cpu] {
                var result = BenchmarkResult(mode: mode)
                do {
                    let detector = try YOLODetector(info: model, computeMode: mode)
                    result.loadMs = detector.loadTime * 1000
                    result.stats = try detector.benchmark(iterations: iterations, warmup: 5)
                } catch {
                    result.error = error.localizedDescription
                }
                result.breakdown = await ComputePlanInspector.breakdown(modelURL: model.url, mode: mode)
                let finished = result
                await MainActor.run { self.benchmarkResults.append(finished) }
            }
            await MainActor.run { self.isBenchmarking = false }
            self.camera.videoQueue.async {
                self.isPaused = false
                self.lastFrameTime = 0
            }
        }
    }
}

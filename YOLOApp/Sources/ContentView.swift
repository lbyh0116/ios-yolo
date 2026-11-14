import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = DetectorViewModel()
    @State private var showBenchmark = false
    /// Lets the preview go clean: tap the image (or the eye button) to bring the panels back.
    @AppStorage("showControls") private var showControls = true
    @AppStorage("previewFillMode") private var fillModeRaw = PreviewFillMode.fill.rawValue
    /// Eye button position as fractions of the window, so it survives rotation and resizing.
    @AppStorage("eyeX") private var storedEyeX = 0.95
    @AppStorage("eyeY") private var storedEyeY = 0.06
    /// Live position while dragging; only written back to storage when the finger lifts.
    @State private var draggingEye: CGPoint?

    private var fillMode: PreviewFillMode { PreviewFillMode(rawValue: fillModeRaw) ?? .fill }

    var body: some View {
        GeometryReader { geo in
            let landscape = geo.size.width > geo.size.height
            // Landscape keeps the phone look but shrinks the card so it never covers the whole frame.
            let panelWidth: CGFloat = landscape
                ? min(max(geo.size.width * 0.38, 340), 460)
                : geo.size.width - 24
            let eye = eyePoint(in: geo.size)
            // The card goes to whichever side the eye button is not on, so they can't overlap.
            let panelOnRight = eye.x < geo.size.width / 2

            ZStack {
                Color.black.ignoresSafeArea()

                preview
                    .ignoresSafeArea()

                if showControls {
                    if landscape {
                        floatingCluster(width: panelWidth, onRight: panelOnRight)
                    } else {
                        phoneChrome
                    }
                }

                eyeButton(in: geo.size, at: eye)
            }
            .animation(.easeInOut(duration: 0.18), value: showControls)
        }
        .onAppear { viewModel.start() }
        .sheet(isPresented: $showBenchmark) {
            BenchmarkView(viewModel: viewModel)
        }
    }

    // MARK: - Preview

    private var preview: some View {
        ZStack {
            CameraPreview(
                session: viewModel.camera.session,
                device: viewModel.device,
                fillMode: fillMode,
                onRotationAngle: { viewModel.updateRotationAngle($0) }
            )
            DetectionOverlay(
                detections: viewModel.detections,
                tags: viewModel.tags,
                imageSize: viewModel.imageSize,
                fillMode: fillMode
            )
        }
        .contentShape(Rectangle())
        .onTapGesture {
            withAnimation(.easeInOut(duration: 0.18)) { showControls.toggle() }
        }
    }

    /// Portrait / squarish windows: exactly the phone layout — HUD on top, card across the bottom.
    private var phoneChrome: some View {
        VStack(spacing: 0) {
            statsHUD
                .padding(.horizontal, 12)
                .padding(.top, 8)
                .allowsHitTesting(false)
            Spacer(minLength: 0)
            controlsPanel
                .padding(.horizontal, 12)
                .padding(.bottom, 8)
        }
    }

    /// Landscape: HUD and card share one width and sit together in the corner opposite the eye button.
    private func floatingCluster(width: CGFloat, onRight: Bool) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            statsHUD
                .allowsHitTesting(false)
            controlsPanel
        }
        .frame(width: width, alignment: .leading)
        .padding(12)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: onRight ? .bottomTrailing : .bottomLeading)
    }

    private var controlsPanel: some View {
        controlsContent
            .padding(12)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Eye button

    private func eyePoint(in size: CGSize) -> CGPoint {
        let fraction = draggingEye ?? CGPoint(x: storedEyeX, y: storedEyeY)
        return CGPoint(
            x: min(max(fraction.x, 0.04), 0.96) * size.width,
            y: min(max(fraction.y, 0.04), 0.96) * size.height
        )
    }

    private func eyeButton(in size: CGSize, at point: CGPoint) -> some View {
        Image(systemName: showControls ? "eye.slash" : "eye")
            .font(.system(size: 13, weight: .semibold))
            .padding(9)
            .background(.ultraThinMaterial, in: Circle())
            .contentShape(Circle())
            .opacity(showControls ? 0.55 : 0.85)
            .position(point)
            .onTapGesture {
                withAnimation(.easeInOut(duration: 0.18)) { showControls.toggle() }
            }
            .gesture(
                DragGesture(minimumDistance: 6)
                    .onChanged { value in
                        draggingEye = CGPoint(
                            x: min(max(value.location.x / size.width, 0.04), 0.96),
                            y: min(max(value.location.y / size.height, 0.04), 0.96)
                        )
                    }
                    .onEnded { _ in
                        if let dragged = draggingEye {
                            storedEyeX = dragged.x
                            storedEyeY = dragged.y
                        }
                        draggingEye = nil
                    }
            )
            .accessibilityLabel(showControls ? "隐藏参数界面" : "显示参数界面")
    }

    // MARK: - HUD

    private var statsHUD: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(viewModel.detectionMode == .aprilTag ? "AprilTag" : viewModel.selectedModel?.name ?? "无模型")
                    .font(.headline)
                Text(viewModel.detectionMode == .aprilTag ? "CPU" : viewModel.computeMode.title)
                    .font(.caption.bold())
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.accentColor, in: Capsule())
                Spacer()
                Text(String(format: "%.1f FPS", viewModel.stats.fps))
                    .font(.headline.monospacedDigit())
            }
            if viewModel.detectionMode.runsYOLO {
                HStack(spacing: 12) {
                    metric("推理", String(format: "%.1f ms", viewModel.stats.inferenceMs))
                    metric("后处理", String(format: "%.2f ms", viewModel.stats.postprocessMs))
                    metric("目标", "\(viewModel.stats.objectCount)")
                }
            }
            if viewModel.detectionMode.runsAprilTag {
                HStack(spacing: 12) {
                    metric("AprilTag", String(format: "%.1f ms", viewModel.stats.tagMs))
                    metric("Tag", "\(viewModel.stats.tagCount)")
                    metric("内参", viewModel.stats.intrinsicsFromCamera ? "相机标定" : "FOV 估算")
                }
            }
            if let error = viewModel.errorMessage {
                Text(error).font(.caption).foregroundColor(.red)
            } else if let status = viewModel.statusMessage {
                Text(status).font(.caption).foregroundColor(.secondary)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 14))
    }

    private func metric(_ title: String, _ value: String) -> some View {
        HStack(spacing: 4) {
            Text(title).foregroundColor(.secondary)
            Text(value).monospacedDigit()
        }
        .font(.caption)
    }

    // MARK: - Controls

    private var controlsContent: some View {
        VStack(spacing: 12) {
            Picker("检测模式", selection: $viewModel.detectionMode) {
                ForEach(DetectionMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            if viewModel.detectionMode.runsYOLO {
                yoloControls
            }
            if viewModel.detectionMode.runsAprilTag {
                aprilTagControls
            }

            HStack(spacing: 8) {
                Menu {
                    ForEach(CaptureFormatGroup.groups(of: viewModel.formats)) { group in
                        Section(group.title) {
                            ForEach(group.options) { option in
                                Button {
                                    viewModel.selectFormat(option)
                                } label: {
                                    if option.id == viewModel.activeFormatID {
                                        Label(option.menuTitle, systemImage: "checkmark")
                                    } else {
                                        Text(option.menuTitle)
                                    }
                                }
                            }
                        }
                    }
                } label: {
                    Label(viewModel.formatLabel, systemImage: "camera.aperture")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(viewModel.formats.isEmpty)

                Picker("画面", selection: $fillModeRaw) {
                    ForEach(PreviewFillMode.allCases) { mode in
                        Text(mode.title).tag(mode.rawValue)
                    }
                }
                .pickerStyle(.segmented)
                .frame(width: 120)
            }

            HStack {
                Menu {
                    ForEach(ModelGroup.groups(of: viewModel.models)) { group in
                        Section(group.title) {
                            ForEach(group.models) { model in
                                Button {
                                    viewModel.selectedModel = model
                                } label: {
                                    if model == viewModel.selectedModel {
                                        Label(model.menuTitle, systemImage: "checkmark")
                                    } else {
                                        Text(model.menuTitle)
                                    }
                                }
                            }
                        }
                    }
                } label: {
                    Label(viewModel.selectedModel?.name ?? "选择模型", systemImage: "cube.box")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)

                Button {
                    showBenchmark = true
                } label: {
                    Label("跑分对比", systemImage: "speedometer")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
            }

            Button {
                withAnimation(.easeInOut(duration: 0.18)) { showControls = false }
            } label: {
                Label("隐藏参数界面", systemImage: "eye.slash")
                    .font(.caption)
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.plain)
            .foregroundColor(.secondary)
        }
        .disabled(viewModel.isLoading || viewModel.isBenchmarking)
    }

    @ViewBuilder
    private var yoloControls: some View {
        Picker("计算单元", selection: $viewModel.computeMode) {
            ForEach(ComputeMode.allCases) { mode in
                Text(mode.title).tag(mode)
            }
        }
        .pickerStyle(.segmented)

        Text(viewModel.computeMode.detail)
            .font(.caption2)
            .foregroundColor(.secondary)

        HStack {
            Text("置信度").font(.caption)
            Slider(value: $viewModel.confidence, in: 0.05...0.95, step: 0.05)
            Text(String(format: "%.2f", viewModel.confidence))
                .font(.caption.monospacedDigit())
                .frame(width: 36)
        }

        Toggle(isOn: $viewModel.stickFigure) {
            Label("火柴人（只追踪人体）", systemImage: "figure.walk")
                .font(.caption)
        }
        .toggleStyle(.switch)

        if viewModel.stickFigure {
            Text(viewModel.selectedModel?.isPose == true
                 ? "pose 模型只输出人体 17 个关键点，画骨架不画其他类别"
                 : "当前模型没有关键点输出，换个 pose 模型才会出骨架")
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }

    @ViewBuilder
    private var aprilTagControls: some View {
        HStack {
            Stepper(value: $viewModel.tagSizeCM, in: 1...100, step: 0.1) {
                Text(String(format: "Tag 边长 %.1f cm", viewModel.tagSizeCM))
                    .font(.caption.monospacedDigit())
            }
            Picker("图案方向", selection: $viewModel.tagOrientation) {
                ForEach(TagPatternOrientation.allCases) { orientation in
                    Text(orientation.title).tag(orientation)
                }
            }
            .pickerStyle(.segmented)
            .frame(width: 120)
        }

        HStack {
            Text("平滑").font(.caption)
            Slider(value: $viewModel.tagSmoothing, in: 0...1)
            Text(viewModel.tagSmoothing < 0.34 ? "灵敏" : viewModel.tagSmoothing < 0.67 ? "均衡" : "稳定")
                .font(.caption)
                .frame(width: 36)
        }

        Text("tag36h11 · 原点=Tag 左上角 · X→右 Y→上 Z→垂直指出 Tag · 坐标为相机系 (cm)")
            .font(.caption2)
            .foregroundColor(.secondary)
    }
}

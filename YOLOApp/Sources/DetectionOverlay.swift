import SwiftUI

/// Maps normalized frame coordinates onto the preview of that frame.
struct PreviewMapping {
    enum Mode {
        case fill
        case fit
    }

    private let shown: CGSize
    private let origin: CGPoint

    init(imageSize: CGSize, viewSize: CGSize, mode: Mode) {
        guard imageSize.width > 0, imageSize.height > 0 else {
            shown = .zero
            origin = .zero
            return
        }
        let horizontal = viewSize.width / imageSize.width
        let vertical = viewSize.height / imageSize.height
        let scale = mode == .fill ? max(horizontal, vertical) : min(horizontal, vertical)
        shown = CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
        origin = CGPoint(x: (viewSize.width - shown.width) / 2, y: (viewSize.height - shown.height) / 2)
    }

    func point(_ p: CGPoint) -> CGPoint {
        CGPoint(x: origin.x + p.x * shown.width, y: origin.y + p.y * shown.height)
    }

    func rect(_ r: CGRect) -> CGRect {
        CGRect(origin: point(r.origin), size: CGSize(width: r.width * shown.width, height: r.height * shown.height))
    }
}

/// Draws YOLO boxes, stick-figure skeletons and AprilTag poses over the camera preview of the same frame.
struct DetectionOverlay: View {
    let detections: [Detection]
    let tags: [TagOverlay]
    let imageSize: CGSize
    let fillMode: PreviewFillMode

    /// COCO 17-keypoint skeleton, in the order Ultralytics' pose heads emit them.
    private static let skeletonEdges: [(Int, Int)] = [
        (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    ]

    var body: some View {
        Canvas { context, size in
            guard imageSize.width > 0, imageSize.height > 0 else { return }
            let mapping = PreviewMapping(imageSize: imageSize, viewSize: size, mode: fillMode == .fill ? .fill : .fit)

            for detection in detections {
                let rect = mapping.rect(detection.rect)
                let color = Color.forClass(detection.classIndex)
                if let keypoints = detection.keypoints {
                    // Stick-figure mode: the box only anchors the figure, the skeleton carries the signal.
                    context.stroke(Path(roundedRect: rect, cornerRadius: 4), with: .color(color.opacity(0.35)), lineWidth: 1.5)
                    drawSkeleton(keypoints, mapping: mapping, color: color, in: &context)
                    let label = "\(detection.label) \(Int(detection.confidence * 100))%"
                    drawLabel(label, at: CGPoint(x: rect.minX, y: rect.minY), above: true, color: color.opacity(0.85), in: &context)
                } else {
                    context.stroke(Path(roundedRect: rect, cornerRadius: 4), with: .color(color), lineWidth: 2.5)
                    let label = "\(detection.label) \(Int(detection.confidence * 100))%"
                    drawLabel(label, at: CGPoint(x: rect.minX, y: rect.minY), above: true, color: color, in: &context)
                }
            }

            for tag in tags {
                drawTag(tag, mapping: mapping, in: &context)
            }
        }
        .allowsHitTesting(false)
    }

    private static let axisColors: [(name: String, color: Color)] = [("X", .red), ("Y", .green), ("Z", .blue)]

    /// Draws the 19 skeleton bones plus a dot per visible keypoint; low-confidence points are skipped
    /// so occluded limbs don't flicker across the screen.
    private func drawSkeleton(_ keypoints: [Keypoint], mapping: PreviewMapping, color: Color, in context: inout GraphicsContext) {
        let threshold: Float = 0.35
        let points: [CGPoint?] = keypoints.map { keypoint in
            keypoint.confidence >= threshold ? mapping.point(keypoint.point) : nil
        }

        for (from, to) in Self.skeletonEdges {
            guard from < points.count, to < points.count, let start = points[from], let end = points[to] else { continue }
            var bone = Path()
            bone.move(to: start)
            bone.addLine(to: end)
            context.stroke(bone, with: .color(color), style: StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
        }

        for point in points.compactMap({ $0 }) {
            let dot = CGRect(x: point.x - 2.5, y: point.y - 2.5, width: 5, height: 5)
            context.fill(Path(ellipseIn: dot), with: .color(.white))
        }
    }

    private func drawTag(_ tag: TagOverlay, mapping: PreviewMapping, in context: inout GraphicsContext) {
        var layer = context
        layer.opacity = tag.isStale ? 0.35 : 1

        if tag.trail.count > 1 {
            var trail = Path()
            trail.addLines(tag.trail.map(mapping.point))
            layer.stroke(trail, with: .color(.yellow.opacity(0.7)), style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
        }

        let quad = tag.quad.map(mapping.point)
        var outline = Path()
        outline.addLines(quad)
        outline.closeSubpath()
        layer.fill(outline, with: .color(.cyan.opacity(0.15)))
        layer.stroke(outline, with: .color(.cyan), lineWidth: 3)

        let origin = mapping.point(tag.origin)
        for (end, axis) in zip(tag.axisEnds.map(mapping.point), Self.axisColors) {
            var line = Path()
            line.move(to: origin)
            line.addLine(to: end)
            layer.stroke(line, with: .color(axis.color), style: StrokeStyle(lineWidth: 4, lineCap: .round))
            let text = layer.resolve(Text(axis.name).font(.system(size: 14, weight: .heavy)).foregroundColor(axis.color))
            layer.draw(text, at: end, anchor: .center)
        }
        layer.fill(Path(ellipseIn: CGRect(x: origin.x - 5, y: origin.y - 5, width: 10, height: 10)), with: .color(.white))

        let p = tag.position * 100
        let distance = (tag.position * tag.position).sum().squareRoot() * 100
        let label = String(format: "ID %ld · %.1f cm\n(%.1f, %.1f, %.1f)", tag.id, distance, p.x, p.y, p.z)
        let center = CGPoint(
            x: quad.map(\.x).reduce(0, +) / CGFloat(max(quad.count, 1)),
            y: quad.map(\.y).reduce(0, +) / CGFloat(max(quad.count, 1))
        )
        drawLabel(label, at: center, above: false, color: .cyan, in: &layer)
    }

    /// Draws text on a colored tag; `above` places it above `point` (box captions), otherwise centered on it.
    private func drawLabel(_ string: String, at point: CGPoint, above: Bool, color: Color, in context: inout GraphicsContext) {
        let text = context.resolve(
            Text(string)
                .font(.system(size: 13, weight: .semibold).monospacedDigit())
                .foregroundColor(.black)
        )
        let textSize = text.measure(in: CGSize(width: 400, height: 80))
        let background = above
            ? CGRect(x: point.x, y: max(point.y - textSize.height - 4, 0), width: textSize.width + 8, height: textSize.height + 4)
            : CGRect(x: point.x - textSize.width / 2 - 4, y: point.y - textSize.height / 2 - 2,
                     width: textSize.width + 8, height: textSize.height + 4)
        context.fill(Path(roundedRect: background, cornerRadius: 4), with: .color(color))
        context.draw(text, at: CGPoint(x: background.minX + 4, y: background.minY + 2), anchor: .topLeading)
    }
}

extension Color {
    static func forClass(_ index: Int) -> Color {
        Color(hue: Double((index * 47) % 360) / 360, saturation: 0.8, brightness: 0.95)
    }
}

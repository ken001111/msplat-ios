import SwiftUI
import UIKit
import QuartzCore
import Msplat

// MARK: - Pixel conversion (iOS UIImage)

func pixelDataToUIImage(_ pd: PixelData) -> UIImage {
    let w = pd.width, h = pd.height
    let n = w * h
    let rgba = UnsafeMutablePointer<UInt8>.allocate(capacity: n * 4)
    pd.pixels.withUnsafeBufferPointer { src in
        for i in 0..<n {
            rgba[i * 4]     = UInt8(min(max(src[i * 3], 0), 1) * 255)
            rgba[i * 4 + 1] = UInt8(min(max(src[i * 3 + 1], 0), 1) * 255)
            rgba[i * 4 + 2] = UInt8(min(max(src[i * 3 + 2], 0), 1) * 255)
            rgba[i * 4 + 3] = 255
        }
    }
    let data = Data(bytesNoCopy: rgba, count: n * 4, deallocator: .custom { ptr, _ in
        ptr.deallocate()
    })
    let provider = CGDataProvider(data: data as CFData)!
    let cgImg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                        bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                        bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                        provider: provider, decode: nil, shouldInterpolate: false,
                        intent: .defaultIntent)!
    return UIImage(cgImage: cgImg)
}

// MARK: - Memory probe

func currentResidentBytes() -> UInt64 {
    var info = mach_task_basic_info()
    var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
    let kerr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
        }
    }
    return kerr == KERN_SUCCESS ? info.resident_size : 0
}

func fmtBytes(_ b: UInt64) -> String {
    let mb = Double(b) / 1_048_576
    return String(format: "%.0f MB", mb)
}

// MARK: - Engine

@MainActor
final class Engine: ObservableObject {
    @Published var status: String = "tap Start"
    @Published var image: UIImage?
    @Published var iteration: Int = 0
    @Published var totalIterations: Int = 200
    @Published var splatCount: Int = 0
    @Published var lastMsPerStep: Float = 0
    @Published var lastLoss: Float = 0
    @Published var residentMB: UInt64 = 0
    @Published var running: Bool = false

    func start(datasetPath: String, iterations: Int) {
        running = true
        totalIterations = iterations
        status = "loading dataset…"

        Thread.detachNewThread { [weak self] in
            guard let self else { return }
            autoreleasepool {
                let dataset = GaussianDataset(path: datasetPath)
                var config = TrainingConfig()
                config.iterations = iterations
                config.numDownscales = 0
                config.bgColor = (0, 0, 0)
                let trainer = GaussianTrainer(dataset: dataset, config: config)

                DispatchQueue.main.async { self.status = "training" }

                var batchStart = CACurrentMediaTime()
                var batchSteps = 0

                for i in 0..<iterations {
                    let stats = trainer.step()
                    batchSteps += 1

                    let shouldReport = (i % 10 == 0) || (i == iterations - 1)
                    if shouldReport {
                        let batchEnd = CACurrentMediaTime()
                        let avgMs = Float((batchEnd - batchStart) / Double(batchSteps) * 1000.0)
                        let pd = trainer.render(cameraIndex: 0)
                        let img = pixelDataToUIImage(pd)
                        let iter = stats.iteration
                        let count = stats.splatCount
                        let loss = stats.loss
                        let rss = currentResidentBytes()

                        DispatchQueue.main.async {
                            self.image = img
                            self.iteration = iter
                            self.splatCount = count
                            self.lastMsPerStep = avgMs
                            self.lastLoss = loss
                            self.residentMB = rss / 1_048_576
                        }
                        batchStart = CACurrentMediaTime()
                        batchSteps = 0
                    }
                }

                DispatchQueue.main.async {
                    self.status = "done"
                    self.running = false
                }
            }
        }
    }
}

// MARK: - UI

struct ContentView: View {
    @StateObject private var engine = Engine()
    @State private var iterations: Int = 200

    private var datasetPath: String? {
        Bundle.main.url(forResource: "Scan_recent", withExtension: nil)?.path
    }

    var body: some View {
        VStack(spacing: 12) {
            // Render preview
            ZStack {
                Color.black
                if let img = engine.image {
                    Image(uiImage: img)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                }
                if engine.running && engine.iteration == 0 {
                    ProgressView().tint(.white).scaleEffect(1.5)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .cornerRadius(8)

            // Stats
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("status:").bold(); Text(engine.status)
                    Spacer()
                    Text("\(engine.residentMB) MB resident")
                        .font(.system(.caption, design: .monospaced))
                        .foregroundStyle(.secondary)
                }
                ProgressView(value: Double(engine.iteration), total: Double(engine.totalIterations))
                HStack {
                    Text("iter \(engine.iteration) / \(engine.totalIterations)")
                    Spacer()
                    Text("\(engine.splatCount) splats")
                    Spacer()
                    Text(String(format: "%.1f ms/step", engine.lastMsPerStep))
                    Spacer()
                    Text(String(format: "loss=%.5f", engine.lastLoss))
                }
                .font(.system(.caption, design: .monospaced))
            }

            // Controls
            HStack {
                Stepper(value: $iterations, in: 50...2000, step: 50) {
                    Text("iters: \(iterations)")
                }
                Spacer()
                Button("Start") {
                    guard let path = datasetPath else { return }
                    engine.start(datasetPath: path, iterations: iterations)
                }
                .buttonStyle(.borderedProminent)
                .disabled(engine.running || datasetPath == nil)
            }

            if datasetPath == nil {
                Text("⚠️ Scan_recent folder not in app bundle — add it via Xcode (see README).")
                    .foregroundStyle(.red).font(.caption)
            }
        }
        .padding()
    }
}

@main
struct IpadTestApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}

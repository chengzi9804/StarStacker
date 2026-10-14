import SwiftUI
import PhotosUI
import Photos
import CoreGraphics
import UIKit

struct ContentView: View {
    @State private var selectedItems: [PhotosPickerItem] = []
    @State private var sourceImages: [CGImage] = []
    @State private var resultImage: UIImage? = nil
    @State private var isProcessing = false
    @State private var isLoadingPhotos = false
    @State private var progress: Double = 0
    @State private var sigmaClipThreshold: Double = 2.0
    @State private var useAlignment: Bool = true
    @State private var errorMessage: String?

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 20) {
                    headerView

                    PhotosPicker(selection: $selectedItems, maxSelectionCount: 60, matching: .images) {
                        Label("选择星空照片 (最多60张)", systemImage: "photo.badge.plus")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(.blue.gradient)
                            .foregroundColor(.white)
                            .cornerRadius(12)
                    }

                    if isLoadingPhotos {
                        ProgressView("正在加载照片...")
                    } else if !sourceImages.isEmpty {
                        Text("已加载 \(sourceImages.count) 张照片")
                            .foregroundColor(.secondary)
                    }

                    settingsCard

                    if let error = errorMessage {
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                            .multilineTextAlignment(.center)
                    }

                    Button(action: runStack) {
                        if isProcessing {
                            ProgressView(value: progress) {
                                Text("处理中... \(Int(progress * 100))%")
                            }
                            .frame(maxWidth: .infinity)
                        } else {
                            Label("开始 Sigma-Clip 堆栈", systemImage: "wand.and.stars")
                                .frame(maxWidth: .infinity)
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                    .disabled(sourceImages.isEmpty || isProcessing || isLoadingPhotos)

                    if let result = resultImage {
                        resultSection(result)
                    }
                }
                .padding()
            }
            .navigationTitle("StarStacker")
        }
        .onChange(of: selectedItems) { _, newItems in
            Task { await loadImages(from: newItems) }
        }
    }

    private var headerView: some View {
        VStack(spacing: 4) {
            Image(systemName: "moon.stars.fill")
                .font(.system(size: 40))
                .foregroundColor(.indigo)
            Text("星空像素堆栈")
                .font(.title2.bold())
            Text("Sigma-Clip 降噪 · 多帧对齐 · Metal GPU加速")
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding(.top)
    }

    private var settingsCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Sigma 阈值")
                Spacer()
                Text(String(format: "%.1fσ", sigmaClipThreshold))
                    .foregroundColor(.blue)
            }
            Slider(value: $sigmaClipThreshold, in: 1.0...5.0, step: 0.5)

            Toggle("启用星点对齐", isOn: $useAlignment)
        }
        .padding()
        .background(.gray.opacity(0.1))
        .cornerRadius(12)
    }

    private func resultSection(_ image: UIImage) -> some View {
        VStack {
            Text("堆栈结果")
                .font(.headline)
            Image(uiImage: image)
                .resizable()
                .scaledToFit()
                .frame(maxHeight: 300)
                .cornerRadius(8)
                .shadow(radius: 5)

            Button(action: saveToPhotoLibrary) {
                Label("保存到相册", systemImage: "square.and.arrow.down")
            }
            .buttonStyle(.bordered)
        }
    }

    @MainActor
    private func loadImages(from items: [PhotosPickerItem]) async {
        isLoadingPhotos = true
        sourceImages.removeAll()
        errorMessage = nil
        resultImage = nil

        for item in items {
            do {
                guard let data = try await item.loadTransferable(type: Data.self),
                      let ui = UIImage(data: data),
                      let cg = ui.cgImage else { continue }
                sourceImages.append(cg)
            } catch {
                errorMessage = "部分照片加载失败: \(error.localizedDescription)"
            }
        }

        isLoadingPhotos = false
    }

    private func runStack() {
        isProcessing = true
        progress = 0
        errorMessage = nil

        Task {
            do {
                let engine = try StackEngine()
                let result = try await engine.stack(
                    images: sourceImages,
                    sigma: Float(sigmaClipThreshold),
                    align: useAlignment,
                    onProgress: { p in
                        DispatchQueue.main.async { self.progress = p }
                    }
                )
                DispatchQueue.main.async {
                    self.resultImage = UIImage(cgImage: result)
                    self.isProcessing = false
                }
            } catch {
                DispatchQueue.main.async {
                    self.errorMessage = "处理失败: \(error.localizedDescription)"
                    self.isProcessing = false
                }
            }
        }
    }

    private func saveToPhotoLibrary() {
        guard let image = resultImage else { return }
        PHPhotoLibrary.shared().performChanges {
            PHAssetCreationRequest.creationRequestForAsset(from: image)
        } completionHandler: { success, error in
            DispatchQueue.main.async {
                if !success {
                    errorMessage = "保存失败: \(error?.localizedDescription ?? "未知错误")"
                }
            }
        }
    }
}

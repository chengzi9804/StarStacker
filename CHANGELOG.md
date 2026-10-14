# 更新日志 (Changelog)

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/) 格式。

## [Unreleased]

### 新增
- **高像素堆栈模式**：多帧同场景照片自动对齐后平均降噪（降噪量 ≈ √N），叠加 USM 锐化，放大后细节更清晰
- 模式切换：星空堆栈 / 高像素堆栈， segmented control 一键切换
- 高像素模式使用干净平均输出（无星空专用的对比度拉伸/阴影提亮），再走 3×3 unsharp mask（amount=0.35）
- 新增 Metal kernel：`averageOutput`、`sharpenOutput`
- 新增 `StackMode` 枚举

### 修复
- Metal `accumulateWithShift` 中 write-only 纹理 `.read()` 编译错误 → 改 `access::read_write`
- `MTLRegionMake2D` 误用 CGRect → 改为 4 个 Int 参数
- `makeComputePipelineState` 在 Xcode 16 SDK 中为 async → 加 `await`
- `withCheckedThrowingContinuation` 在 async throws 上下文需 `try`
- 缺少 AppIcon.appiconset 导致 asset catalog 失败
- `xcodebuild -exportArchive` 无 Team 失败 → 改为手动打包 Payload/ → zip 成 IPA

## [0.1.0] - 2026-10-14

### 新增
- 初始版本：星空 Sigma-Clip 多帧堆栈
- Metal GPU 计算引擎：累积、降采样 SSD 对齐（±2px on /8 网格 = ±16px full res）
- Sigma-Clip 输出：均值 + 轻度对比度拉伸 + 阴影提亮
- SwiftUI 界面：PhotosPicker（最多 60 张）、Sigma 滑块、对齐开关、进度条、结果预览、保存相册
- GitHub Actions CI：macos-15 runner 自动编译 → 归档（无签名）→ 手动打包未签名 IPA → upload artifact
- xcodegen 工程生成

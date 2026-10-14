# StarStacker

iOS 星空像素堆栈 App — Sigma-Clip 多帧降噪 + Metal GPU 加速 + 星点对齐。

## 功能
- 从相册选择多张星空照片（最多 60 张）
- Metal GPU 多帧累加（float32 精度）
- 暴力搜索星点对齐（±8 像素，在 GPU 上算 SSD）
- Sigma-Clip 统计降噪
- 一键保存结果到相册

## 要求
- iOS 17.0+
- A15 / M 系列芯片（完整 Metal 计算管线）

## GitHub Actions 自动构建
1. Fork 本仓库到你的 GitHub
2. 推送到 main 分支，或在 Actions 页面手动触发
3. 构建完成后从 Artifacts 下载 `StarStacker-Unsigned-IPA`
4. 用 [Sideloadly](https://sideloadly.io/) 或 AltServer 重签侧载到 iPhone

## 本地构建（需要 Mac）
```bash
brew install xcodegen
bash scripts/bootstrap.sh
xcodegen generate
open StarStacker.xcodeproj
```

## 技术栈
- SwiftUI
- Metal Compute (Sigma-Clip + SSD alignment)
- ImageIO / CoreGraphics
- xcodegen（无 CocoaPods / SwiftPM 依赖）

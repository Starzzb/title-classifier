具体使用说明在 readme.md，FAQ 在 FAQ.md

常用命令：

  # 启动 GUI
  uv run title-classifier gui

  # 扫描目录，生成待审 CSV
  uv run title-classifier scan -d "F:\Videos"

  # 扫描单个文件
  uv run title-classifier scan -d "F:\Videos\video.mp4"

  # 追加模式（不覆盖已有 CSV）
  uv run title-classifier scan -d "F:\Videos" -a

  # 全盘索引：仅同步数据库，不生成 CSV（完成后自动生成待视觉识别清单）
  uv run title-classifier scan -d "F:\" --sync-db --exclude-dir "$RECYCLE.BIN" "System Volume Information"

  # AI 优化标题（可选）
  uv run title-classifier refine -p gcli

  # 音频识别生成字幕（可选）
  uv run title-classifier audio -p mimo

  # 视觉识别（YOLO + VLM；有 NVIDIA GPU 自动启用 CUDA 加速）
  uv run title-classifier vision --use-yolo -p gcli

  # 视觉识别（全面分析，detect+pose+segment 三模型投票）
  uv run title-classifier vision --use-yolo --comprehensive -p gcli

  # 使用 CLIP 预分类
  uv run title-classifier vision --use-clip --use-yolo -p gcli

  # 重试之前失败的行
  uv run title-classifier vision --use-yolo --retry-failed -p gcli

  # 预览重命名（不执行）
  uv run title-classifier rename --dry-run

  # 执行重命名
  uv run title-classifier rename

  # 数据库
  uv run title-classifier db stats
  uv run title-classifier db search --query "关键词"
  uv run title-classifier db import --all

  # 模型下载
  uv run python scripts/download_models.py      # 一键下载全部
  uv run python scripts/download_clip.py        # 仅 CLIP
  uv run python scripts/download_yolo_models.py # 仅 YOLO

  # 设备基准（CPU/OpenVINO vs CUDA/PyTorch，附加速比）
  uv run python scripts/bench_yolo_device.py "视频.mp4" --frames 30

  # 重建虚拟环境（torch/torchvision 自动装 cu130 版，无 GPU 平台自动回退 CPU 版）
  uv sync

常用参数：

  全局:
    -v, --verbose      详细输出（控制台 DEBUG 日志）
    --log              日志文件路径（默认 logs/<日期>/）

  scan:
    -d, --dir            目标目录或文件（必需）
    -o, --output         输出 CSV 路径
    --output-dir         输出目录（默认 data/output）
    -a, --append         追加模式
    --exclude-dir        排除目录（可多个）
    --force              强制重新分类（全量 CSV + 更新数据库）
    --sync-db            仅同步数据库（不生成 CSV，适合全盘索引）
    --deep               深度同步
    --exclude-images     同步时排除图片

  refine:
    -c, --csv          CSV 文件路径
    -p, --provider     AI Provider（gcli/zhipu/mimo/siliconflow/ollama）

  audio:
    -c, --csv          CSV 文件路径
    -p, --provider     AI Provider（默认 mimo）
    --all              处理所有未识别文件

  vision:
    -c, --csv          CSV 文件路径
    -p, --provider     AI Provider
    --use-yolo         启用 YOLO 检测
    --comprehensive    全面分析（detect+pose+segment 三模型投票）
    --use-clip         启用 CLIP 预分类
    --yolo-conf        YOLO 置信度阈值（默认 0.5）
    --clip-threshold   CLIP 置信度阈值（默认 0.25）
    --vlm-frames       VLM 帧数（默认 10）
    --analysis-step    采样间隔秒数（默认 5.0）
    --max-sample-frames 采样帧上限（默认 50，超出均匀分布）
    --device           推理设备 auto/cuda/cpu（默认读配置；auto=有GPU(显存>=4GB)用CUDA）
    --backend          YOLO 后端 auto/openvino/pytorch（默认 auto；CUDA 设备自动用 pytorch）
    --concurrent       并发处理视频数（默认 4；GPU 下推理自动串行，收益在抽帧/解码重叠）
    --no-motion-detection  禁用运动检测前置过滤（默认启用）
    --motion-threshold 运动检测阈值%（默认 10.0）
    --no-scene-detection   禁用场景切分（默认启用，>60s 视频有效）
    --scene-threshold  场景切分阈值 0-1，越小越敏感（默认 0.3）
    --max-scenes       最大场景数（默认 10）
    --frames-per-scene 每场景抽帧数（默认 10）
    --all              处理所有未识别文件
    --retry-failed     重试 vision_failed=true 的行
    --auto-import      完成后自动导入数据库
    --debug            调试模式（保存 VLM 请求等中间产物）
    --debug-dir        调试输出目录（默认 data/debug）

  rename:
    -c, --csv          CSV 文件路径
    --dry-run          模拟运行
    --use-rclone       用 rclone 传输（适合远程盘）
    --rclone-path      rclone 可执行文件路径
    --max-workers      重命名并发线程数

  db:
    db init | db import [--csv x | --all] | db list [--limit n]
    db search [--query 关键词 | --tag 标签 | --source 来源]
    db show <id> | db history <id> | db stats

配置：
  .env                       — API Key（已加入 .gitignore）
    GCLI_API_KEY             — gcli API key（视觉识别，推荐）
    MIMO_API_KEY             — MiMo API key（视觉+音频）
    ZHIPU_API_KEY            — 智谱 API key（文本优化）
    SILICONFLOW_API_KEY      — 硅基流动（视觉识别，多模型可选）
  config/default.toml        — 默认配置（YOLO/CLIP/VAD/场景切分等）
  config/user.toml           — 个人覆盖配置（provider 选择/设备/主题；已入库）
  config/providers.json      — 自定义 Provider

模型目录：
  models/clip/               — CLIP 模型
  models/yolo/               — YOLO 模型（detect/pose/segment + openvino 缓存）

推理设备（v8.3.0 起）：

  默认 auto：有 NVIDIA GPU（显存>=4GB）→ CUDA（YOLO 8.8x / CLIP 38x）；
  无 GPU → CPU + OpenVINO。GPU 推理自动串行（吞吐 ~49 帧/s，超过 CPU 8 线程并发
  的 ~21 帧/s），并发 worker 的收益在帧抽取/解码/VLM 调用重叠。
  详见 FAQ「CUDA加速」章节；复测基准：uv run python scripts/bench_yolo_device.py

帧选择策略（分区段覆盖）：

  视频按采样间隔（默认5秒）提取帧（上限50帧），再选出 vlm_frames 帧发送给 VLM。
  选帧采用"分区段"策略，将采样帧等分为 vlm_frames 个区段，每个区段内独立选最优帧，
  保证全视频均匀覆盖，不会因后半段置信度低而被忽略。

  示例：60秒视频，采样12帧，选10帧
    区段1 [帧0-1]  → 内部比评分 → 选1帧（覆盖 0-6s）
    区段2 [帧2-3]  → 内部比评分 → 选1帧（覆盖 6-12s）
    ...
    区段10 [帧10-11] → 内部比评分 → 选1帧（覆盖 50-60s）

  区段内评分规则（启用 CLIP 时）：
    置信度 30% + 关键点可见性 20% + 姿态变化 20% + CLIP帧差异度 30%
  无 CLIP 时降级为：置信度 40% + 关键点 30% + 姿态变化 30%
  无人体帧 → 取区段中间帧（保证覆盖）

字幕上下文优化：

  发送给 VLM 的字幕上下文按字幕时间段去重：多个帧落入同一字幕区间时合并显示，
  避免重复发送相同字幕内容。

  示例：
    - 图1,2,3,4@4.0s-30.0s: [00:00:01 --> 00:00:31] 字幕内容...
    - 图9@48.0s: [00:00:33 --> 00:00:59] 字幕内容...

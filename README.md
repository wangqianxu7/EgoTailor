# EgoTailor

基于 [X-LeBench](https://github.com/VvV7/X-LeBench) 流程，构建 **单人 21 天 × 每天 8 小时** 的第一人称日常作息数据集。

## 项目结构

```
EgoTailor/
├── generation/           # 数据集生成流水线
│   ├── config.py
│   ├── schedule_templates.py   # Stage 1: persona + 日程模板
│   ├── retrieve_videos.py      # Stage 3: 视频检索
│   └── build_lifelog.py        # 主入口
├── output/               # 生成的 lifelog 数据
│   ├── persona/
│   ├── days/
│   └── lifelog/
└── notebooks/            # Ego4D 数据探索
```

Ego4D 视频元信息复用 `../X-LeBench/generation/ego4d_info/video_info.json`。

## 快速开始

```bash
cd /root/EgoTailor
python -m generation.build_lifelog        # 默认 seed=42
python -m generation.build_lifelog 123    # 自定义随机种子
```

## 数据集特点

- **1 人**：美国 ENFP 软件工程师
- **21 天 v2 日程**：2026-01-05 起，含工作日变体、周末主题轮换、异常事件
- **每天 8 小时视频**，场景覆盖 indoor / outdoor / mixed
- **连续精确时间戳**：`start_timestamp` / `end_timestamp`（ISO 8601）
- **Caption**：来自 `video_info.json` 的 `consolidated_summary`

### 日程 v2 亮点

| 类型 | 内容 |
|------|------|
| 工作日变体 | 周一周会、周二深度工作、周四项目汇报、周五团队聚餐+酒吧 |
| 周末主题 | 周六轮换：远足 / 骑行+市集 / 户外探险；周日轮换：电影马拉松 / cozy 宅家 / 游戏日 |
| 随机波动 | 起床 ±15min、通勤 ±10min、锻炼类型/时长随机 |
| 异常事件 | Day 3 雨天、Day 8 出差、Day 11 加班、Day 16 临时约会、Day 17 看医生 |
| 晚间多样化 | 看剧 / 打游戏 / 读书 / 早睡 / 音乐 / 偶尔 video call |

## 配置

编辑 `generation/config.py` 可调整天数、时长、起始日期、场景配额等。

---

## 兴趣/偏好分析（Hierarchical RAG + vLLM MLLM）

基于 21 天 lifelog，通过 **clip（分钟级）→ hour（小时级）→ day（日级）→ period（多日级）** 四层 RAG 检索，结合 vLLM 多模态接口分析用户兴趣与习惯。

```bash
cd /root/EgoTailor

# 1. 构建分层索引
python -m analysis.run_analysis build-index

# 2. 预览 RAG 检索（无需 vLLM）
python -m analysis.run_analysis retrieve -q "outdoor hiking exercise"

# 3. 自动挖掘兴趣点（需 vLLM 服务）
export VLLM_API_BASE=http://localhost:8000/v1
export VLLM_MODEL=Qwen2-VL-7B-Instruct
python -m analysis.run_analysis mine-interests --auto

# 4. 指定主题查询
python -m analysis.run_analysis mine-interests -q "cooking food preferences"

# 5. 分析单个 video_uid（自动加载本地 Ego4D 视频帧）
python -m analysis.run_analysis analyze-clip --video-uid 40b86eb9-a408-4119-ba7c-402b050be506

# 7. 映射 lifelog video_uid -> Ego4D 原视频
python -m analysis.run_analysis build-registry

# 8. RAG + Ego4D 原视频 + vLLM 细粒度行为偏好分析（当前服务端口 8080）
python -m analysis.run_analysis vlm-profile \
  -q "outdoor hiking exercise" "food cooking eating" "work meeting office" "home leisure tv games" \
按视频时长**自适应**均匀抽帧（整条 clip 共 N 张，不是 N fps）：

| 时长 | 抽帧数 |
|------|--------|
| < 5 分钟 | 4–8 帧 |
| 5–20 分钟 | 8–16 帧 |
| 20–30 分钟 | 16 帧 |
| > 30 分钟 | 16–32 帧 |

```bash
python -m analysis.run_analysis vlm-profile --max-clips 2          # 自适应
python -m analysis.run_analysis vlm-profile --frames 12            # 强制固定 12 帧/clip
export FRAMES_PER_CLIP=12                                          # 环境变量固定帧数
```

# 9. 全量主题分析
python -m analysis.run_analysis vlm-profile --auto --max-clips 2
```

环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_API_BASE` | `http://127.0.0.1:8080/v1` | vLLM OpenAI 兼容接口 |
| `VLLM_MODEL` | `Qwen3-VL-8B-Instruct` | 多模态模型名 |
| `EGO4D_VIDEO_ROOT` | `/mnt/data_oss/raw_data/Ego4d/v2/full_scale` | `{video_uid}.mp4` 目录 |

输出：
- `output/analysis/interest_report.json` — RAG 文本分析
- `output/analysis/video_registry.json` — lifelog clip → `{video_uid}.mp4` 映射
- `output/analysis/vlm_behavior_profile.json` — RAG + VLM 细粒度行为偏好报告

# EgoTailor

基于 [X-LeBench](https://github.com/VvV7/X-LeBench) 流程，构建 **单人 30 天 × 每天 8 小时** 的第一人称日常作息数据集。

与 X-LeBench 的关键差异：**人物画像是从语料反推出来的，不是先写死再去检索**。
先统计 Ego4D 实际有什么，再描述一个"能产生这些素材的人"，日程只从这份配额里取。
结果是 100% 的时间槽都能配到场景匹配的真实视频（`T1_scenario_match`），且全程不复用任何一条 clip。

## 项目结构

```
EgoTailor/
├── generation/                    # 数据集生成流水线
│   ├── config.py                  # 天数、时长、检索阈值、路径
│   ├── persona_generator.py       # Stage 1a: 从语料推导人物画像 + 场景配额
│   ├── schedule_from_quota.py     # Stage 1b: 把配额排进一天的叙事骨架
│   ├── retrieve_videos.py         # Stage 3: 视频检索
│   ├── build_lifelog.py           # 主入口
│   ├── persona_quota.py           # 手写配额表（对照基线，--handwritten）
│   └── schedule_templates.py      # 旧版日程模板（对照基线，--legacy）
├── analysis/                      # 分层 RAG + VLM 行为分析
├── scripts/                       # 抽帧、拼接、批量 VLM 分析
└── output/
    ├── persona/portrait_<id>.json           # 完整人物画像 + 配额证据
    ├── days/<persona_id>/day_NN_<date>.json # 每日 plan，逐槽对齐视频
    └── lifelog/lifelog_<id>_<N>d.json       # 全量数据（下游脚本读这个）
```

Ego4D 视频元信息使用**本地已下载子集**：`generation/ego4d_info/video_info.json`
（从 X-LeBench 全量 `video_info.json` 中筛出 `/mnt/data_oss/raw_data/Ego4d/v2/full_scale` 里实际存在的 `{video_uid}.mp4`）。

## 快速开始

```bash
cd /root/EgoTailor

# （可选）按本地 mp4 重新筛选 video_info
python scripts/filter_local_video_info.py

# 生成 30 天 lifelog
python -m generation.build_lifelog          # 默认 seed=42
python -m generation.build_lifelog 7        # 换 seed = 换一个人

# 只看画像，不跑检索
python -m generation.persona_generator 42
python -m generation.persona_generator 42 --json
```

**换 seed 会得到不同的人**，因为角色是按语料供给量抽出来的：

| seed | 人物 | 场景数 | 时长 |
|------|------|--------|------|
| 42 | 居家 + 营生 + 手作 | 25 | 219h |
| 7  | 研究 + 居家 | 25 | 213h |
| 11 | 营生 + 园艺 | 22 | 163h |
| 3  | 手作 + 居家 | 24 | 224h |

## 数据集特点

- **人物画像**：由 `persona_generator` 从语料推导。配额表的每一行都带自己的证据
  （候选片数 / 无复用上限 / 中位时长），改了 pool 过滤条件重跑即可，不存在过期的手写数字。
- **30 天日程**：2026-01-05 起，工作日/周末不同骨架，5 个异常日（雨天、赶稿、来客、外出、病假）。
- **两个正交旋钮**：改配额 → 这个人做什么、做多频；改 `schedule_from_quota.py` 的
  `DAY_SHAPES` → 什么时候做、什么顺序。
- **每槽一条视频**：能对上 plan 起点就对齐；若上一段超时，下一段紧接上一段结束时间。
- **Caption**：来自 `video_info.json` 的 `consolidated_summary`。

### 护栏

`persona_generator.check()` 在生成时强制以下三条，`build_lifelog` 会把违规打印出来：

| 护栏 | 阈值 | 防的是什么 |
|------|------|-----------|
| 无超额 | 每格需求 ≤ 候选数 | 需要复用 clip |
| 单场景占比 | ≤ 20% | 塌成"全天做饭" |
| 场景熵 | ≥ 4.0 bits | 人格单调 |

**Ego4D 的支撑上限约 35 天**（每天约 26 个槽、不复用）。超过后薄供给的格子先枯竭，
分配器只能倒向深供给场景，熵跌破护栏。30 天在安全区（熵 4.14）。
要更长只能放开 `ALLOW_CROSS_DAY_REUSE` 或降低熵下限——都是数据集设计决策。

### 对照基线

```bash
python -m generation.schedule_from_quota --handwritten   # 手写配额表
python -m generation.build_lifelog 42 --legacy           # 旧版 schedule_templates
```

旧版直接用场景写日程，因此会向语料索要它没有的素材（`Bus` 要 23 次、`Video call` 要 13 次，
供给都是 0），只有约 40% 的槽能配到场景匹配的 clip。保留仅作对照。

## 配置

编辑 `generation/config.py` 调整天数（`TOTAL_DAYS`）、时长、起始日期、检索阈值。
改完直接重跑，配额会按新参数重新推导。

---

## 兴趣/偏好分析（Hierarchical RAG + vLLM MLLM）

基于 lifelog，通过 **clip（分钟级）→ hour（小时级）→ day（日级）→ period（多日级）** 四层 RAG 检索，结合 vLLM 多模态接口分析用户兴趣与习惯。

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
  -q "outdoor hiking exercise" "food cooking eating" "work meeting office" "home leisure tv games"

# 9. 全量主题分析
python -m analysis.run_analysis vlm-profile --auto --max-clips 2
```

抽帧按视频时长**自适应**均匀抽取（整条 clip 共 N 张，不是 N fps）：

| 时长 | 抽帧数 |
|------|--------|
| < 5 分钟 | 4–8 帧 |
| 5–20 分钟 | 8–16 帧 |
| 20–30 分钟 | 16 帧 |
| > 30 分钟 | 16–32 帧 |

```bash
python -m analysis.run_analysis vlm-profile --max-clips 2   # 自适应
python -m analysis.run_analysis vlm-profile --frames 12     # 强制固定 12 帧/clip
export FRAMES_PER_CLIP=12                                   # 环境变量固定帧数
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
- `output/analysis/full_<N>d_vlm/` — 批量 VLM 行为画像（目录名带天数，避免不同长度的运行互相覆盖）

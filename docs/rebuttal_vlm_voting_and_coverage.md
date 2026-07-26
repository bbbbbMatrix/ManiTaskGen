# Rebuttal 操作手册：VLM-voting 统计 & 任务覆盖率

> 分支：`haina-rebuttal-nips2026`
> 适用需求：`demand/rebuttal_demand.md`
> 相关代码：
> - Outcome-based 投票：`src/core/outcome_based_task_generation.py`、`src/scripts/02b_gen_outcome_based_tasks.py`
> - 覆盖率：`src/core/task_coverage_analyzer.py`、`src/scripts/05_coverage_analysis.py`
> - 配置：`config/default_config.yml`、`src/utils/config_manager.py`

两件事互相独立，可分别运行：

| 模块 | 做什么 | 是否需要 key |
|---|---|---|
| Part 1 — VLM-voting | 对一批 raw outcome 任务用多个 VLM 投票，统计 0/1/2/3 分分布，产出可人工审查的包 + 最终取用任务 | 需要 OpenRouter key + Sapien 渲染 |
| Part 2 — 任务覆盖率 | 从生成的 process 任务里抽样，统计 item / receptacle 覆盖率，并与 GPT taskgen 对比 | 不需要 key |

---

## 0. 前置准备

```bash
cd /mnt/hdd_1/haina/workspace/manitaskgen/ManiTaskGen
git checkout haina-rebuttal-nips2026
conda activate manitaskgen          # 或你自己的环境名
python -m pip install pytest        # 跑单测用；已装可跳过
```

跑一遍单测确认环境正常（共 17+ 用例）：

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

所有命令都从仓库根目录执行，并带 `PYTHONPATH=.`（仓库用 `src.*` 包内导入）。

---

## Part 1 — Outcome-based VLM-voting 统计

### 1.1 它在做什么

1. 从 pattern 文件读任务模板（如 `data/templates/manitask_ot200.txt`，或你挑的子集）。
2. 每个 pattern 生成最多 `task_num_per_pattern` 个候选 raw 任务（去重）。
3. 每个 raw 任务用 `vlm_list` 里的每个 VLM 各投一次票（`Feasible` / `Partially feasible` / `Not feasible`）。
4. **分数 = 投 `Feasible` 的 VLM 个数**，取值 `0/1/2/3`（默认 3 个 VLM）。
5. **最终取用 = 分数 ≥ `keep_min_score`**（默认 2 = 多数票）。
6. 产出：0/1/2/3 直方图、每个 raw 任务的分数与各 VLM 判决、投票图、HTML 审查页、最终取用任务。

### 1.2 配置（`config/default_config.yml`）

> **重点**：这次修了一个 bug——以前改 yml 的 `vlm_list` / `task_num_per_pattern` 不生效（getter 返回的是写死的默认值）。现在改 yml 真正生效了。

必填：
```yaml
common:
  open_router:
    api_key: Bearer sk-or-v1-你的真实key
```

投票相关旋钮（`stage2b_outcome_task_generation.outcome_based_task` 下）：
```yaml
stage2b_outcome_task_generation:
  manitaskot_pattern_file: data/templates/manitask_ot200.txt   # 指向你的子集文件即可
  outcome_based_task:
    task_num_per_pattern: 5      # 每个 pattern 生成几个候选
    keep_min_score: 2            # 分数 ≥ 此值才取用（默认多数票）
    vlm_list:                    # 投票用的 VLM
      - "openai/gpt-4.1"
      - "anthropic/claude-3.5-haiku"
      - "google/gemini-2.5-flash-lite-preview-06-17"
```

**挑 pattern 子集**（建议，控制成本）：把选中的 pattern 复制到一个新文件，例如 `data/templates/manitask_ot200_subset.txt`，然后把上面的 `manitaskot_pattern_file` 指过去。不挑就跑全部 200 个（很贵）。

**先廉价验通**（可选）：把 `vlm_list` 临时只留 1 个便宜模型、`task_num_per_pattern` 调成 1，跑通流程后再恢复全量。

### 1.3 运行

```bash
CONFIG_FILE=config/default_config.yml bash scripts/run_02b_gen_outcome_tasks.sh
```

### 1.4 产物解读

- **`runs/output/outcome_based_task.txt`** —— 最终取用的任务（每行一条；路径 = yml 里 `outcome_based_task_txt_save_path`，契约保持不变）。
- **`runs/output/outcome_review/vote_results.json`** —— 核心，结构：
  - `histogram`: `{"0": n, "1": n, "2": n, "3": n}` —— **这就是你要的 0/1/2/3 分分布**。
  - `keep_min_score`、`vlm_list`：本次用的阈值与模型。
  - `tasks`: 所有 raw 任务，每条含 `task_id`、`task_description`、`pattern`、`score`、`verdicts`（每个 VLM 的判决）、`feasible`、`image_dir`、引用的 `platforms`/`objects`。
  - `kept_tasks`: 取用的任务子集。
- **`runs/output/outcome_review/review_gallery.html`** —— 用浏览器打开：按分数 0/1/2/3 分组、内嵌投票图与各 VLM 判决，**用于人工审查质量与 consistency**。
- **`runs/images/image4vote/task_<id>/`** —— 每个任务自己的投票图（已按任务隔离，不会互相覆盖）。

> 想换阈值重算"取用集"不用重跑 VLM：直接读 `vote_results.json`，按 `score >= 你想要的阈值` 过滤 `tasks` 即可。

---

## Part 2 — Process-based 任务覆盖率（vs GPT）

### 2.1 它在做什么

1. 读 ManiTaskGen 生成的 process 任务（`runs/cache/process_based_task.pkl`）。
2. 随机抽 `--sample_size` 个（默认 100），按种子可复现。
3. 把每个任务归一成四类引用：**moving objects**（被搬物体）、**anchor objects**（参照物）、**target platforms**（目标 receptacle）、**source platforms**（来源 receptacle）。
4. 对每个维度统计：每个实例出现次数、覆盖到的不同实例数 / 场景总数、比例。
5. 同样地吃一份 GPT 产出的任务 JSON（按 `demand/example.md` 的 schema），做并排对比。

### 2.2 数据前提（重要）

当前缓存的 `runs/cache/process_based_task.pkl` **只有 2 个 TaskChain**，抽样会变成 `min(100, 2) = 2`，覆盖率没意义。要先重新生成足够多的任务：

在 `config/default_config.yml` 把 `stage2a` 的上限调大：
```yaml
stage2a_process_task_generation:
  process_based_task:
    max_task_num: 200        # 至少 ≥ 你要抽样的数量
```
然后重新生成：
```bash
CONFIG_FILE=config/default_config.yml bash scripts/run_02a_gen_process_tasks.sh
```

### 2.3 （可选）产出 GPT 任务用于对比

按 `demand/example.md`：把 `runs/scene_export_apt_0/scene_input.md` 整段粘给 GPT，把 `images/` 里对应图作为附件上传；再单独发 `INPUT_STAGE=GENERATE_TASKS` + 配额让它出 task。把返回的 JSON 存成例如 `runs/gpt_taskgen/gpt_tasks.json`。

### 2.4 运行

```bash
PYTHONPATH=. python src/scripts/05_coverage_analysis.py \
  --manitaskgen_pkl runs/cache/process_based_task.pkl \
  --scene_graph_pkl runs/cache/scene_graph.pkl \
  --sample_size 100 --seed 0
  # 对比 GPT 时再加：
  # --gpt_json runs/gpt_taskgen/gpt_tasks.json
```

或用 wrapper（环境变量传参）：
```bash
MANITASKGEN_PKL=runs/cache/process_based_task.pkl \
SCENE_GRAPH_PKL=runs/cache/scene_graph.pkl \
SAMPLE_SIZE=100 SEED=0 \
[ GPT_JSON=runs/gpt_taskgen/gpt_tasks.json ] \
bash scripts/run_05_coverage_analysis.sh
```

### 2.5 产物解读（`runs/output/coverage/`）

- **`coverage_report.md`** —— 人看的并排表：

  | Dimension | ManiTaskGen covered/total (ratio) | GPT covered/total (ratio) |
  |---|---|---|
  | moving_objects | 12/68 (0.176) | … |
  | anchor_objects | … | … |
  | target_platforms | … | … |
  | source_platforms | … | … |

  下方还有每方的 per-instance 计数。
- **`coverage_report.json`** —— 机器可读：`meta`（sample_size/seed/场景总数）、`manitaskgen`、`gpt`，每个维度含 `counts`（per-instance 出现次数）、`distinct_covered`、`total`、`ratio`、`uncovered`。

场景总数（分母）来自 `scene_graph.pkl`：movable objects = sensible platform 上的直接子物体；platforms = sensible platform 列表。ManiTaskGen 与 GPT 用**同一组分母**，所以对比公平。（已验证 apt_0 场景 = 68 objects / 60 platforms，与导出 manifest 一致。）

---

## 排错 / 备注

- **改 yml 不生效？** 现在已修复。确认你改的是 `stage2b_outcome_task_generation.outcome_based_task` 下的字段，且用 `CONFIG_FILE=...` 指定了该 yml。运行后看 `latest_config/used_config.yaml` 确认实际加载值。
- **Part 1 太贵/太慢？** 调小 `vlm_list`（甚至 1 个）、`task_num_per_pattern`，或用 pattern 子集文件。这些都是 yml 旋钮。
- **覆盖率抽样太少？** 是 process 任务 pkl 太小（见 2.2），重新生成更多任务即可，不是覆盖率代码的问题。
- **命名空间**：覆盖率按实例 `.name` 匹配。`SceneExporter` 导出的 id 与 scene graph 同名空间，所以 ManiTaskGen 与 GPT 可直接比。若将来 rename_dict 不一致导致名字漂移，实例级匹配会偏低——届时可在 `task_coverage_analyzer.py` 加一层 category 聚合（当前未实现，属已知鲁棒性缺口）。
- **跑测试**：`PYTHONPATH=. python -m pytest tests/ -v`。

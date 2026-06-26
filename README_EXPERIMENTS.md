# SCB5 Zero-Shot — Experiment Pipeline & Reproduction

Supplementary notes for the experiment pipeline. See REPRODUCIBILITY.md and README.md for canonical instructions.

## 一键复现入口（审稿友好）
- 复现指南：`REPRODUCIBILITY.md`
- 一键脚本：`bash reproduce_paper.sh --mode quick --gpu 0`
- 引用模板：`CITATION.cff`

## 主要脚本与职责
- `run_experiment.py` — 底层可复用函数：模型加载、单次零样本评估、`run_single_experiment` API。
- `exp_runner.py` — 论文主流水线（E1/E2 主表格、E4 CAPE 主协议），调用 `run_experiment.run_single_experiment`，输出到 `results/`。
- `master_benchmark.sh` / `master_benchmark_parallel.sh` — [LEGACY] 并行化大规模基准（shard），每个 shard 生成 `results_parallel/<shard>/benchmark_*.json`，最终合并为 `results_parallel/benchmark_final_*.json`（供绘图使用）。
- `cape_robustness.py` — CAPE 的提示词变体鲁棒性探针（A/B/Mix），输出到 `results_robustness/`；并定义 `CAPE_A`/`CAPE_B`。
- `run_revision_experiments.py` — 修订/答审实验（R1–R4，对应论文 E7–E10）：线性探针、bootstrap CI、盲提示、严格多标签评估，输出到 `scb5_zeroshot/results_revision/`。
- `paper/generate_paper_figures.py` — 读取合并基准（`results_parallel/benchmark_final_*.json`）并生成论文图表到 `scb5_zeroshot/paper/figures/`。

## 数据与检查点布局
- 数据集：`datasets_scb/`（YOLO 格式：`images/{train,val,test}` + `labels/{train,val,test}`）。
- 模型权重：`scb5_zeroshot/ckpts/<model>/*`（每个模型一个子目录）。

## 主要结果路径（谁写、用途）
- `scb5_zeroshot/results/` — `exp_runner.py` 的主结果（例如 `e1e2_*.json`, `e4_cape_*.json`, `paper_all_*.json`）。
- `scb5_zeroshot/results_parallel/<shard>/` — 并行 shard 的中间与最终 `benchmark_*.json`（每个 shard 由 `master_benchmark.sh` 写入）。
- `scb5_zeroshot/results_parallel/benchmark_final_*.json` — `master_benchmark_parallel.sh` 合并后的总表（`paper/generate_paper_figures.py` 的输入）。
- `scb5_zeroshot/results_robustness/` — `cape_robustness.py` 的输出（E4 表格来源之一）。
- `scb5_zeroshot/results_revision/` — `run_revision_experiments.py` 的输出（R1–R4 / E7–E10）。
- `scb5_zeroshot/paper/figures/` — 论文图表（PDF/PNG），由 `generate_paper_figures.py` 生成。

## 实验处理流程（高层）
1. 由 `exp_runner.py` 或 `benchmark_runner` 构建 prompt 组（`label_only`, `simple`, `action`, `detailed`）与 CAPE prompts（K=3）。
2. 从 `ckpts/` 加载模型（`clip`/`open_clip`），构建 `encode_image` / `encode_text`。
3. 对每个类将该类的 prompts 编码、L2 归一并均值池化得到类文本向量 `t_c`。
4. 批量编码图片特征 `v`，计算相似度 `v @ t_c^T`，得到 top-1/top-3 预测。
5. 计算指标：Hit@1/Hit@3、混淆矩阵、Macro-F1、Balanced-Acc 等并保存为 JSON。
6. 线性探针（R1 / E7）：提取训练/验证特征，标准化，训练 LogisticRegression（单标签/one-vs-rest），计算 Sample-F1 等。
7. Bootstrap CI（R2 / E8）：基于逐样本预测做 1000 次重采样，计算 Hit@1 与 Macro-F1 的 95% CI。
8. 盲提示对比（R3 / E9）：评估 CAPE_A / CAPE_B / CAPE_C 并保存结果。
9. 多标签评估（R4 / E10）：基于相似度阈值与 top-k 策略，计算 Sample/Macro/Micro F1。
10. 并行基准写入 shard `benchmark_final_*.json`，最后合并为 `results_parallel/benchmark_final_*.json`（绘图输入）。

## 复现实验（常用命令）
- 论文主流水线（单卡、当前目录为仓库根）：
```bash
python scb5_zeroshot/exp_runner.py --gpu 0
```
- 并行运行完整基准（示例，4 shards、GPU 0..3）：
```bash
MODEL_SHARDS="clip;laion;siglip;eva02,dfn" GPU_SHARDS="0;1;2;3" \
  bash scb5_zeroshot/master_benchmark_parallel.sh
```
- 运行 CAPE 鲁棒性实验（E4 细节）：
```bash
python scb5_zeroshot/cape_robustness.py --gpu 0
```
- 运行修订实验（R1–R4 / E7–E10）：
```bash
python scb5_zeroshot/run_revision_experiments.py --gpu 0 --exp r1 r2 r3 r4
```
- 生成/重生成论文图表：
```bash
python scb5_zeroshot/paper/generate_paper_figures.py
```

## 可重复性与注意事项
- CAPE mix 采样与 bootstrap 使用固定随机种子（见脚本内 seed），主流程确定性高。\
- 如果要重跑大规模并行基准，请确保 `datasets_scb/` 与 `scb5_zeroshot/ckpts/` 可访问并设置合适的 `PYTHON_BIN`/`CUDA_VISIBLE_DEVICES`。

---
如需我把 README 扩展为更详细的运行说明（例如示例 `nohup` 启动、资源估算、或生成 `systemd` 单元），我可以继续追加。

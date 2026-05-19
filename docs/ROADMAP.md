# Roadmap

## Thesis Direction

Current thesis direction:

> 面向弱监督标签与伪变化抑制的湿地遥感变化检测方法研究

The repository is no longer organized around a single preselected backbone such as Mamba-CLIP. The current workflow is: build reliable weak-label data, reproduce comparable baselines in one framework, then select and improve a model for wetland pseudo-change suppression.

## Stage 1. Data Layer

- [x] Build six wetland study areas.
- [x] Prepare Sentinel-2 2018/2022 images.
- [x] Read GLC_FCS30D 2018/2022 land-cover products.
- [x] Build initial weak change labels from land-cover differences.
- [x] Add ESA WorldCover 2021 as an auxiliary consistency source.
- [x] Build confidence layers: high-confidence change, high-confidence unchanged, and low-confidence samples.
- [x] Slice wetland train/val/test patches.
- [x] Complete data-quality acceptance table.
- [x] Confirm the data can support weak-label construction, pseudo-change analysis, wetland experiments, and cross-region generalization.

## Stage 2. Unified Training Framework

- [x] Unify dataset reading for wetland, SECOND, and HRSCD.
- [x] Unify training epochs, input size, learning-rate policy, metrics, and result format through config files.
- [x] Add Siamese U-Net baseline.
- [x] Add training history, metrics, checkpoints, and result-summary outputs.
- [x] Complete local workstation checks: dataloader, single-batch smoke tests, visualization scripts, and short trials.

## Stage 3. Public Dataset Preparation

- [x] Prepare SECOND and verify train/val/test reading.
- [x] Prepare HRSCD small sample for local debugging.
- [x] Rebalance HRSCD sample so train/val/test each include changed patches.
- [ ] On the server, expand HRSCD with the full official archive or a larger official-split sample.

## Stage 4. Baseline Reproduction

- [x] Start SECOND + Siamese U-Net formal baseline on the server.
- [ ] Finish SECOND + Siamese U-Net formal baseline and record final metrics.
- [ ] Generate SECOND baseline curves and visualizations.
- [ ] Reproduce at least two additional baselines on SECOND.
- [ ] Migrate selected baselines to HRSCD.
- [ ] Migrate selected baselines to wetland weak-label data.

Priority baseline candidates:

- `Siamese U-Net`
- `ChangeViT`
- `ChangeMamba` or `CDMamba`
- `MaskCD`
- `BAN` or another foundation-model adaptation baseline

## Stage 5. Wetland Method Development

- [ ] Select the final backbone based on SECOND, HRSCD, and wetland transfer results.
- [ ] Design weak-label noise handling or confidence-weighted training.
- [ ] Design pseudo-change suppression using multi-source consistency, temporal evidence, or semantic constraints.
- [ ] Add boundary or small-patch enhancement if baseline visualizations show boundary weakness.
- [ ] Run ablation experiments.
- [ ] Run cross-region generalization experiments.

## Stage 6. Thesis Writing Outputs

- [ ] Baseline result table.
- [ ] Training curves.
- [ ] Qualitative visualization figures.
- [ ] Weak-label confidence analysis figures.
- [ ] Pseudo-change case analysis.
- [ ] Final comparison and ablation tables.

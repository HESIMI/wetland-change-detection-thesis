# Development Roadmap

## Stage 1. Dataset and Baseline

- [x] Build wetland change detection dataset
- [x] Generate binary and semantic change labels
- [x] Create train / val / test patch splits
- [x] Run `Siamese UNet` baseline
- [x] Build unified training framework for shared data loading, metrics, LR policy, and result format

## Stage 2. Candidate Model Reproduction

- [ ] Reproduce `ChangeMamba`
- [ ] Reproduce `ChangeViT`
- [ ] Reproduce `MaskCD`
- [ ] Reproduce `ChangeCLIP`
- [ ] Register reproduced models into the unified `train.py` framework

## Stage 3. Comparative Experiments

- [ ] Unified evaluation on the wetland dataset
- [ ] Efficiency comparison
- [ ] Cross-region generalization experiment
- [ ] Qualitative visualization

## Stage 4. Thesis Model Development

- [ ] Final backbone selection
- [ ] Semantic enhancement module design
- [ ] Decoder refinement
- [ ] Ablation experiments

## Stage 5. Thesis Writing

- [ ] Experimental chapter consolidation
- [ ] Figures and tables
- [ ] Final model description
- [ ] Conclusion and future work

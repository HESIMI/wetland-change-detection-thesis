# MaskCD Official LEVIR-CD Evaluation

## Purpose

This run evaluates the official MaskCD implementation and official pretrained
LEVIR-CD weights on the official HuggingFace LEVIR-CD cropped dataset. It is
used to separate official-model evidence from the lightweight local prototypes.

## Repository And Model

- Official repository: `D:/桌面/毕业论文/official_repos/MaskCD`
- Paper model: MaskCD, IEEE TGRS 2024
- Dataset: `ericyu/LEVIRCD_Cropped_256`
- Pretrained model: `ericyu/MaskCD_LEVIRCD_Cropped256`
- Test split size: 2048
- Input size: 256 x 256
- Batch size on local RTX 2060: 1

The model structure and pretrained weights are official. Local changes to
`test.py` are compatibility changes only:

- add `--batch-size` so the official model can run on 6GB VRAM;
- use `verification_mode="no_checks"` to avoid a HuggingFace split metadata
  mismatch between `validation` and `val`;
- avoid moving Python list fields through newer `accelerate` internals.

## Result

| Dataset | Model | OA | F1 | Precision | Recall | cIoU |
|---|---|---:|---:|---:|---:|---:|
| LEVIR-CD | MaskCD official pretrained | 0.9903 | 0.9030 | 0.9217 | 0.8851 | 0.8232 |

Command:

```powershell
cd D:/桌面/毕业论文/official_repos/MaskCD
set "HF_HOME=D:/hf_cache"
set "HF_DATASETS_CACHE=D:/hf_cache/datasets"
python test.py --dataset ericyu/LEVIRCD_Cropped_256 --model ericyu/MaskCD_LEVIRCD_Cropped256 --batch-size 1
```

Raw log:

```text
Accuracy=0.9903178811073303
mF1=0.9030413031578064
Precision=0.9217486381530762
Recall=0.8850781321525574
cIoU=0.8232226967811584
```

## Interpretation

This result confirms that the official MaskCD model performs strongly on
LEVIR-CD and explains why the previous lightweight `maskcd_style` prototype
should not be treated as a MaskCD reproduction. The weak prototype result was
caused by simplified query-mask decoding and insufficient mask-level training
constraints, not by a weakness of MaskCD itself.

For the thesis, MaskCD should be listed as an official reproduced/reference
baseline only when using this official repository and official weights or a
full official training run. The lightweight local `maskcd_style` result should
remain an internal ablation/prototype note.

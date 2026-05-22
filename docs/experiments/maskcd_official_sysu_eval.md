# MaskCD Official SYSU-CD Evaluation

## Purpose

This run evaluates the official MaskCD implementation and official pretrained
SYSU-CD weights on the official HuggingFace SYSU-CD dataset. It is used as an
official-model reference result, not as a lightweight local prototype.

## Repository And Model

- Official repository: `D:/桌面/毕业论文/official_repos/MaskCD`
- Paper model: MaskCD, IEEE TGRS 2024
- Dataset: `ericyu/SYSU_CD`
- Pretrained model: `ericyu/MaskCD_SYSU_CD`
- Test split size: 4000
- Input size: 256 x 256
- Batch size on local RTX 2060: 1

The model structure and pretrained weights are official. The batch size is
reduced only to fit the local 6GB RTX 2060 memory. This does not change the
model, weights, dataset split, or metric protocol.

Local runtime compatibility changes in `test.py` are the same as the LEVIR-CD
official evaluation:

- add `--batch-size` so official models can run on limited VRAM;
- use `verification_mode="no_checks"` for HuggingFace split metadata tolerance;
- keep the dataloader outside `accelerator.prepare` to avoid newer
  `accelerate` moving non-tensor list fields.

## Result

| Dataset | Model | OA | F1 | Precision | Recall | cIoU |
|---|---|---:|---:|---:|---:|---:|
| SYSU-CD | MaskCD official pretrained | 0.9232 | 0.8289 | 0.8732 | 0.7889 | 0.7078 |

Command:

```powershell
cd D:/桌面/毕业论文/official_repos/MaskCD
$env:HF_HOME='D:/hf_cache'
$env:HF_DATASETS_CACHE='D:/hf_cache/datasets'
$env:TRANSFORMERS_CACHE='D:/hf_cache/transformers'
python test.py --dataset ericyu/SYSU_CD --model ericyu/MaskCD_SYSU_CD --batch-size 1
```

Raw log:

```text
Accuracy=0.923201322555542
mF1=0.8289063572883606
Precision=0.8732367157936096
Recall=0.7888594269752502
cIoU=0.707805335521698
```

Output predictions:

```text
D:/桌面/毕业论文/official_repos/MaskCD/results/ericyu/MaskCD_SYSU_CD/change_map
```

The folder contains 4000 predicted change maps, matching the SYSU-CD test split
size.

## Interpretation

The official MaskCD SYSU-CD result is clearly lower than the official LEVIR-CD
result, but still strong. This is consistent with SYSU-CD being a more complex
scene-level binary change detection benchmark. For the thesis, this result can
be used as an official MaskCD reference baseline on a second public dataset,
while the local lightweight `maskcd_style` model should remain an internal
prototype only.

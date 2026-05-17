# Weak Label Construction Flow

本文湿地弱监督标签构建流程以 GLC_FCS30D 年度土地覆盖产品和 Sentinel-2 双时相影像为主要输入，先生成初始变化标签，再进行伪变化识别、多源一致性筛选和高低置信样本划分。

```mermaid
flowchart TD
    A[GLC_FCS30D 2018 土地覆盖] --> C[研究区裁剪与网格对齐]
    B[GLC_FCS30D 2022 土地覆盖] --> C
    C --> D[T1 土地覆盖 lc_t1_2018.tif]
    C --> E[T2 土地覆盖 lc_t2_2022.tif]
    D --> F[逐像素差分]
    E --> F
    F --> G[初始变化图 initial_change.tif]

    H[Sentinel-2 2018 生长季合成影像] --> J[光谱变化证据计算]
    I[Sentinel-2 2022 生长季合成影像] --> J
    J --> K[NDVI / NDWI / Brightness 变化得分]
    K --> L[spectral_change_score.tif]

    D --> M[语义类别重编码]
    E --> M
    M --> N[伪变化候选识别]
    N --> O[pseudo_change_mask.tif]

    G --> P[多源一致性筛选]
    L --> P
    P --> Q[multi_source_consistency.tif]

    G --> R[双时相时序一致性代理筛选]
    L --> R
    O --> R
    R --> S[temporal_consistency.tif]

    G --> T[置信样本划分]
    L --> T
    O --> T
    Q --> T
    S --> T

    T --> U[高置信变化 high_confidence_change.tif]
    T --> V[高置信未变化 high_confidence_unchanged.tif]
    T --> W[低置信样本 low_confidence_mask.tif]
    T --> X[置信等级 confidence_score.tif]
```

## Output Layers

初始弱标签：

- `data/weak_labels/initial_change/<area>/lc_t1_2018.tif`
- `data/weak_labels/initial_change/<area>/lc_t2_2022.tif`
- `data/weak_labels/initial_change/<area>/initial_change.tif`

置信筛选结果：

- `data/weak_labels/confidence/<area>/high_confidence_change.tif`
- `data/weak_labels/confidence/<area>/high_confidence_unchanged.tif`
- `data/weak_labels/confidence/<area>/low_confidence_mask.tif`
- `data/weak_labels/confidence/<area>/confidence_score.tif`


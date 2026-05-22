param(
    [int]$Epochs = 20,
    [int]$BatchSize = 1,
    [int]$TrainBatches = 200,
    [int]$ValBatches = 50,
    [int]$TestBatches = 50,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$repo = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repo

$experiments = @(
    @{ Dataset = "levir-cd"; Model = "siamese_unet"; Config = "configs/training/levir_siamese_unet_local.json" },
    @{ Dataset = "levir-cd"; Model = "changeformer"; Config = "configs/training/levir_changeformer_local.json" },
    @{ Dataset = "levir-cd"; Model = "changemamba_lite"; Config = "configs/training/levir_changemamba_lite.json" },
    @{ Dataset = "levir-cd"; Model = "cdmamba_maskcd"; Config = "configs/training/levir_cdmamba_maskcd.json" },

    @{ Dataset = "whu-cd"; Model = "siamese_unet"; Config = "configs/training/whu_siamese_unet_local.json" },
    @{ Dataset = "whu-cd"; Model = "changeformer"; Config = "configs/training/whu_changeformer_local.json" },
    @{ Dataset = "whu-cd"; Model = "changemamba_lite"; Config = "configs/training/whu_changemamba_lite_local.json" },
    @{ Dataset = "whu-cd"; Model = "cdmamba_maskcd"; Config = "configs/training/whu_cdmamba_maskcd.json" },

    @{ Dataset = "sysu-cd"; Model = "siamese_unet"; Config = "configs/training/sysu_siamese_unet_local.json" },
    @{ Dataset = "sysu-cd"; Model = "changeformer"; Config = "configs/training/sysu_changeformer_local.json" },
    @{ Dataset = "sysu-cd"; Model = "changemamba_lite"; Config = "configs/training/sysu_changemamba_lite_local.json" },
    @{ Dataset = "sysu-cd"; Model = "cdmamba_maskcd"; Config = "configs/training/sysu_cdmamba_maskcd.json" }
)

foreach ($exp in $experiments) {
    $runName = "local_reduced_$($exp.Dataset)_$($exp.Model)"
    $outDir = "runs/$runName"
    Write-Host "=== Running $runName ==="
    & $Python -m src.wetland_cd.training.train `
        --config $exp.Config `
        --epochs $Epochs `
        --batch-size $BatchSize `
        --limit-train-batches $TrainBatches `
        --limit-val-batches $ValBatches `
        --limit-test-batches $TestBatches `
        --outdir $outDir
}

Write-Host "All local reduced experiments finished."

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
    @{ Model = "siamese_unet"; Config = "configs/training/levir_siamese_unet_local.json" },
    @{ Model = "changeformer"; Config = "configs/training/levir_changeformer_local.json" },
    @{ Model = "changemamba_lite"; Config = "configs/training/levir_changemamba_lite.json" },
    @{ Model = "cdmamba_maskcd"; Config = "configs/training/levir_cdmamba_maskcd.json" }
)

foreach ($exp in $experiments) {
    $runName = "local_reduced_levir-cd_$($exp.Model)"
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

Write-Host "LEVIR-CD local reduced experiments finished."

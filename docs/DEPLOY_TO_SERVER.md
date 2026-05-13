# Server Deployment

## Project summary

- Training code lives in `training/`
- Processed dataset lives in `data/processed/`
- Current processed data size is about `6.07 GB`
- Existing experiment outputs live in `runs/`

## Recommended server target path

Use a dedicated project directory such as:

```bash
mkdir -p ~/wetland_cd_project
```

## Upload from local Windows machine

Open PowerShell in `D:\桌面\文献` and run:

```powershell
scp -r .\项目 hesimin@202.121.140.53:~/wetland_cd_project
```

If you only want the minimum runnable training package, upload this instead:

```powershell
scp -r .\项目\training .\项目\data\processed .\项目\requirements.txt .\项目\DEPLOY_TO_SERVER.md hesimin@202.121.140.53:~/wetland_cd_project/
```

You will be prompted for the SSH password:

```text
hsm2003
```

## After login on the server

```bash
ssh hesimin@202.121.140.53
cd ~/wetland_cd_project/项目
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate wetland_cd
python -m pip install -r requirements.txt
python training/inspect_dataset.py
python training/train_baseline.py --epochs 5 --batch-size 4 --outdir runs/siamese_unet_server
```

## One-command server run

After upload, you can also run:

```bash
cd ~/wetland_cd_project/项目
bash training/run_server_baseline.sh
```

## Notes

- `training/train_baseline.py` now uses project-relative default paths, so it works on Linux.
- `training/dataset.py` now remaps old Windows manifest paths to the current project directory automatically.
- Because `dataset_manifest.csv` still contains Windows absolute paths, keep the current folder layout intact after upload.

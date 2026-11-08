@echo off
setlocal
cd /d E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel
echo [bat] start %DATE% %TIME% > "E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/flowmatching/runs_author/pilot_status.txt"

D:/anaconda/envs/difftumor/python.exe -u flowmatching/train_seg_author.py --arm real --synt_model ddpm --max_epochs 300 --batch_size 2 --optim_lr 0.0002 --val_every 100 --cache_rate 0.5 --cache_mode disk --workers 4 --save_checkpoint --max_iters 0 --max_val_iters 0 --logdir "E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/flowmatching/runs_author/real" > "E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/flowmatching/runs_author/pilot_real.log" 2>&1
if errorlevel 1 (
  echo [bat] real failed %DATE% %TIME% >> "E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/flowmatching/runs_author/pilot_status.txt"
  exit /b 1
)
echo [bat] real done %DATE% %TIME% >> "E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/flowmatching/runs_author/pilot_status.txt"

D:/anaconda/envs/difftumor/python.exe -u flowmatching/train_seg_author.py --arm fm --synt_model fm --max_epochs 300 --batch_size 2 --optim_lr 0.0002 --val_every 100 --cache_rate 0.5 --cache_mode disk --workers 4 --save_checkpoint --max_iters 0 --max_val_iters 0 --fm_steps 8 --logdir "E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/flowmatching/runs_author/fm" > "E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/flowmatching/runs_author/pilot_fm.log" 2>&1
if errorlevel 1 (
  echo [bat] fm failed %DATE% %TIME% >> "E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/flowmatching/runs_author/pilot_status.txt"
  exit /b 1
)
echo [bat] fm done %DATE% %TIME% >> "E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/flowmatching/runs_author/pilot_status.txt"

D:/anaconda/envs/difftumor/python.exe -u flowmatching/train_seg_author.py --arm ddpm --synt_model ddpm --max_epochs 300 --batch_size 2 --optim_lr 0.0002 --val_every 100 --cache_rate 0.5 --cache_mode disk --workers 4 --save_checkpoint --max_iters 0 --max_val_iters 0 --fm_steps 8 --logdir "E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/flowmatching/runs_author/ddpm" > "E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/flowmatching/runs_author/pilot_ddpm.log" 2>&1
if errorlevel 1 (
  echo [bat] ddpm failed %DATE% %TIME% >> "E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/flowmatching/runs_author/pilot_status.txt"
  exit /b 1
)
echo [bat] all done %DATE% %TIME% >> "E:/thesisAndCode/code/DiffTumor/STEP3.SegmentationModel/flowmatching/runs_author/pilot_status.txt"

## cmd line

```bash
cd ./dicomseg

export PYTHONPATH=$(pwd) &&  python3 dicomseg/aneurysm.py \
--ID 01901124_20250617_MR_21404020048 \ 
--InputsDicomDir /mnt/e/pipeline/新增資料夾/01901124_20250617_MR_21404020048/MRA_BRAIN \
--Inputs /mnt/e/pipeline/新增資料夾/01901124_20250617_MR_21404020048/Image_nii/Pred.nii.gz \
--Output_folder /mnt/e/pipeline/新增資料夾/

```

## use function
[aneurysm.py](aneurysm.py)

```python
# root_path xxxx/Deep_Aneurysm/ID
execute_dicomseg_platform_json(_id: int, root_path: str, group_id: int)

```
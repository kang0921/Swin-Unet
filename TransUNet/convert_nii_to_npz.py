import os
import numpy as np
import SimpleITK as sitk
from sh import gunzip
import gzip

# NIfTI格式的影像文件所在的目錄
NII_DIR = "/home/siplab5/Swin-Unet/DATASET/nnUNet_raw/nnUNet_raw_data/Task055_Lung/imagesTr/"

# 得到目錄下的所有文件的路徑
def get_filelist(dir, filelist):
    if os.path.isdir(dir):
        for nii_file in os.listdir(dir):
            newDir = os.path.join(dir, nii_file)
            filelist.append(newDir)
    return filelist


def gunzip_file(file_path):
    if file_path.endswith('.gz'):
        output_path = os.path.splitext(file_path)[0] # 將檔名和副檔名分開
        with gzip.open(file_path, 'rb') as file_in, open(output_path, 'wb') as file_out:
            file_out.write(file_in.read())
        print(f"解壓縮完成: {file_path} -> {output_path}")
    # else:
    #     print(f"不需要解壓縮: {file_path}")


list = get_filelist(NII_DIR, [])    # 獲取 NII_DIR 目錄下的所有文件路徑

for list_file in list:
    gunzip_file(list_file)                          # 解壓縮檔案
    Refimg = sitk.ReadImage(list_file)              # 讀取NIfTI格式的醫學影像
    RefimgArray = sitk.GetArrayFromImage(Refimg)    # 將SimpleITK影像轉換為NumPy 
    fileName = list_file.split('/')[-1]             # 從檔案路徑中獲取檔案的名稱(根據"/"將完整的檔案路徑 list_file 拆分成一個字符串列表並取得列表中的最後一個元素，即檔案名稱)
    fileName = fileName.replace('nii', 'npz')       
    outputDir = f"/home/siplab5/Swin-Unet/DATASET/nnUNet_raw/nnUNet_raw_data/Task055_Lung/imagesTr/{fileName}"
    np.savez(outputDir, vol_data = RefimgArray)     # savez: 儲存多個陣列在一個zip的檔案中

import SimpleITK as sitk
from skimage.util import montage
import matplotlib.pyplot as plt
import numpy as np

def convert_nii_to_png(fileName):

    fileName = "/home/siplab5/Swin-Unet/DATASET/nnUNet_raw/nnUNet_raw_data/Task055_Lung/imagesTr/Subj_4_0000"
    fileType = ".nii.gz"

    img = sitk.ReadImage( fileName + fileType )
    img_arr = sitk.GetArrayFromImage(img)
    print(img_arr.shape)
    
    img_arr_4d = img_arr[np.newaxis, ...]   # 將 3D 陣列轉換為 4D 陣列，添加一個額外的維度
    print(img_arr_4d.shape)

    fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    ax.imshow(montage(img_arr_4d[0]), cmap = 'gray')

    plt.savefig(fileName + ".png")
    plt.close()

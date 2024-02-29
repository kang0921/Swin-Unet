import SimpleITK as sitk
from skimage.util import montage
import matplotlib.pyplot as plt
import numpy as np
import os

def convert_nii_to_png(_name):

    folder = "/home/siplab5/Swin-Unet/TransUNet/model_out/predictions"
    # folder = "/home/siplab5/Swin-Unet/data/Synapse_lung/data_lung/testing_label/"
    # folder = "/home/siplab5/Swin-Unet/data/Synapse_lung/data_lung/testing_image/"
    fileName = os.path.join(folder, _name)
    fileType = ".nii"

    img = sitk.ReadImage( fileName + fileType )
    img_arr = sitk.GetArrayFromImage(img)
    print(img_arr.shape)
    
    img_arr_4d = img_arr[np.newaxis, ...]   # 將 3D 陣列轉換為 4D 陣列，添加一個額外的維度
    print(img_arr_4d.shape)

    fig, ax = plt.subplots(1, 1, figsize = (20, 20))
    ax.imshow(montage(img_arr_4d[0]), cmap = 'gray')

    plt.savefig("/home/siplab5/Swin-Unet/TransUNet/output_png/" + _name + ".png")
    plt.close()

if __name__ == '__main__':
    
    convert_nii_to_png("Subj_1_pred")
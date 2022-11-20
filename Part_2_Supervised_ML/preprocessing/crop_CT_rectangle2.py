import numpy as np
import pandas as pd
import SimpleITK as stk
import random
from scipy import ndimage
from skimage.transform import resize
import matplotlib.pyplot as plt
import nibabel as nib
from scipy import ndimage
from skimage import morphology
import pydicom
from resize_crop_pad import resize_image_with_crop_or_pad, crop_with_mask, resize_ct



df = pd.read_csv("/processing/ertugrul/Part_2/df_426__withlabel_mask.csv")
#df = pd.read_csv("/processing/ertugrul/Part_2/Labels&paths/CNN_791_paths_labels_m_full_20.csv")


CT_paths = df["Image"].to_list()
mask_paths = df["Mask"].to_list()
#scan = nib.load(CT_paths[52])
#ct = scan.get_fdata()
#mask = nib.load(mask_paths[52])
#mask = mask.get_fdata()

#ct = stk.ReadImage(CT_paths[52])
ct = stk.ReadImage(CT_paths[52])
ct= stk.GetArrayFromImage(ct)
#mask = stk.ReadImage(mask_paths[52])
mask = stk.ReadImage(mask_paths[52])
mask= stk.GetArrayFromImage(mask)

cropped_ct_mask = crop_with_mask(mask, ct)
#resized_cropped_ct = resize_image_with_crop_or_pad(cropped_ct_mask, img_size=(190, 270, 96))
#resized_ct = resize_image_with_crop_or_pad(ct, img_size=(252, 252, 96))
#resized_masked_ct = resize_image_with_crop_or_pad(mask*ct, img_size=(316, 280, 96))
resized_cropped_ct1 = resize_ct(cropped_ct_mask, dim=(96, 192, 192))
resized_cropped_ct2 = resize_image_with_crop_or_pad(cropped_ct_mask, img_size=(96, 192, 192))
resized_ct = resize_image_with_crop_or_pad(ct, img_size=(96, 192, 192))
resized_masked_ct = resize_image_with_crop_or_pad(mask*ct, img_size=(96, 192, 192))


print("ct shape:", ct.shape)
print("cropped_ct shape:", cropped_ct_mask.shape)

print("resized_cropped_ct1 shape:", resized_cropped_ct1.shape)
print("resized_cropped_ct2 shape:", resized_cropped_ct2.shape)


dim = [96, 192, 192]
#resized_ct = resize_ct(ct, dim)
#resized_cropped =resize_ct(cropped_ct,dim)
#resized_mask = resize_ct(mask, dim)
#resized_masked_ct = resize_ct(mask*ct, dim)


#ms(cropped_ct)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)
axes = [ax1, ax2,ax3,ax4,ax5, ax6]

axes[0].imshow(np.squeeze(ct)[75, :, :], cmap="gray")
axes[1].imshow(np.squeeze(mask)[75, :, :], cmap="gray")
axes[2].imshow(np.squeeze(ct*mask)[45, :, :], cmap="gray")
axes[3].imshow(np.squeeze(cropped_ct_mask)[45, :, :], cmap="gray")
axes[4].imshow(np.squeeze(resized_cropped_ct1)[45, :, :], cmap="gray")
axes[5].imshow(np.squeeze(resized_cropped_ct2)[45, :, :], cmap="gray")

plt.tight_layout()

plt.show()
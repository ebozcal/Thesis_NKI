# Build model.
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
from multislice import ms
#df = pd.read_csv("/processing/ertugrul/Part_2/Labels&paths/CNN_791_paths_labels_m_full_20.csv")


def resize_image_with_crop_or_pad(image, img_size=(192, 192, 96)):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[slicer], to_padding)

def crop_with_mask(mask_dat, ct_dat):

    for i in range(mask_dat.ndim):
        ct_dat = np.swapaxes(ct_dat, 0, i)  # send ct i-th axis to front
        mask_dat= np.swapaxes(mask_dat, 0, i)  # send mask i-th axis to front
        print("ct_dat_shape", ct_dat.shape)

        while np.all(mask_dat[0] == 0):
            ct_dat = ct_dat[1:]    # Crop CT where all mask values are zero in that axis
            mask_dat = mask_dat[1:]

        while np.all(mask_dat[-1] == 0):
            ct_dat = ct_dat[:-1]  # Crop CT where all mask values are zero in that axis
            mask_dat = mask_dat[:-1]

        ct_dat = np.swapaxes(ct_dat, 0, i)
        mask_dat = np.swapaxes(mask_dat, 0, i)

    return ct_dat

def resize_ct(ct, dim):
    ct  = np.flip(ct)
    max_dim = np.max((ct.shape[1], ct.shape[2]))

    zoom_z = [dim[0] if ct.shape[0] > dim[0] else ct.shape[0]][0]
    zoom_y = [int(np.round(ct.shape[1]*dim[1]/max_dim)) if max_dim > dim[1] else ct.shape[1]][0]
    zoom_x = [int(np.round(ct.shape[2]*dim[2]/max_dim)) if max_dim > dim[2] else ct.shape[2]][0]
    ct = resize(ct, (zoom_z, zoom_y, zoom_x), preserve_range=True)
            
    z_pad = dim[0] - ct.shape[0]
    y_pad = dim[1] - ct.shape[1]
    x_pad = dim[2] - ct.shape[2]
        
    ct = np.pad(ct, ((int(np.floor(z_pad / 2)), int(np.ceil(z_pad / 2))),
        (int(np.floor(y_pad / 2)), int(np.ceil(y_pad / 2))),
        (int(np.floor(x_pad / 2)), int(np.ceil(x_pad / 2)))),
        'constant', constant_values=0)
    return ct


df = pd.read_csv("/processing/ertugrul/Part_2/df_426__withlabel_mask.csv")
#df = pd.read_csv("/processing/ertugrul/Part_2/Labels&paths/CNN_791_paths_labels_m_full_20.csv")


CT_paths = df["Image"].to_list()
mask_paths = df["Mask"].to_list()

#scan = nib.load(CT_paths[52])
#ct = scan.get_fdata()

#mask = nib.load(mask_paths[52])
#mask = mask.get_fdata()

ct = stk.ReadImage(CT_paths[52])
ct= stk.GetArrayFromImage(ct)
mask = stk.ReadImage(mask_paths[52])
mask= stk.GetArrayFromImage(mask)

cropped_ct = crop_with_mask(mask, ct)
cropped_ct = resize_image_with_crop_or_pad(cropped_ct, img_size=(192, 192, 96))
print("cropped_ct shape:", cropped_ct.shape)
dim = [96, 192, 192]
#resized_ct = resize_ct(mask*ct, dim)

#ms(cropped_ct)

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3)
axes = [ax1, ax2,ax3,ax4,ax5, ax6]

axes[0].imshow(np.squeeze(ct)[:, :, 55], cmap="gray")
axes[2].imshow(np.squeeze(cropped_ct)[:, :, 45], cmap="gray")
axes[4].imshow(np.squeeze(mask)[:, :, 55], cmap="gray")
plt.tight_layout()

plt.show()



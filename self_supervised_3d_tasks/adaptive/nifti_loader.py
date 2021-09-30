import nibabel as nib
import numpy as np
from skimage import transform

def nifti_load(path, padding_channel=False, do_cubic_resize=False):
    data = nib.load(path).get_fdata()
    if do_cubic_resize:
        data = transform.resize(data, (128, 128, 128))
        data = np.asarray(data)
    if padding_channel:
        data = np.expand_dims(data, len(data.shape))
    return data

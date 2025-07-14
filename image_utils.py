import numpy as np
import base64
from PIL import Image
import io

def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr)
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - arr_min) / (arr_max - arr_min)
    return (norm * 255).astype(np.uint8)

def rotate_90_cc(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    new_arr = np.zeros((w, h), dtype=arr.dtype)
    for i in range(h):
        for j in range(w):
            new_arr[w - 1 - j, i] = arr[i, j]
    return new_arr

def rotate_90_c(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape
    new_arr = np.zeros((w,h), dtype=arr.dtype)
    for i in range(h):
        for j in range(w):
            new_arr[i, j] = arr[w - 1 - j, i]
    return new_arr

def resize_image(arr: np.ndarray, target_size=(1024, 1024)) -> np.ndarray:
    from PIL import Image
    pil = Image.fromarray(arr)
    resized = pil.resize(target_size, resample=Image.NEAREST if arr.ndim == 2 else Image.BILINEAR)
    return np.array(resized)

def overlay_mask_on_slice(slice_img: np.ndarray, mask: np.ndarray, color=(187, 63, 63)) -> np.ndarray:
    if slice_img.ndim == 2:
        background = np.stack([slice_img]*3, axis=-1)
    else:
        background = slice_img.copy()

    mask = mask.astype(bool)
    overlay = background.copy()
    for i in range(3):
        overlay[..., i] = np.where(
            mask,
            (1 - 0.5) * background[..., i] + 0.5 * color[i],
            background[..., i]
        )

    return overlay.astype(np.uint8)

def array_to_base64_png(arr):
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()
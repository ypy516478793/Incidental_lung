from scipy.ndimage.interpolation import rotate
from PIL import Image, ImageEnhance
import numpy as np

def balance_any_data(X, y):
    cls, size_all_cls = np.unique(y, return_counts=True)
    small_cls = np.argmin(size_all_cls)

    small_size = size_all_cls[small_cls]
    gap = np.abs(size_all_cls[0] - size_all_cls[1])
    repeat_n = np.ceil(gap / small_size).astype(np.int)

    all_ids = np.arange(len(y))
    small_ids = all_ids[y == small_cls]
    large_ids = all_ids[y != small_cls]
    sampled_small_ids = np.random.choice(np.repeat(small_ids, repeat_n), gap, replace=False)
    resampled_ids = np.concatenate([large_ids, small_ids, sampled_small_ids])
    np.random.shuffle(resampled_ids)

    X = X[resampled_ids]
    y = y[resampled_ids]
    return X, y


def augment(sample, ifflip=False, ifrotate=False, ifswap=False, ifcontrast=False, ifbright=False, ifsharp=False):
    #                     angle1 = np.random.rand()*180
    if ifrotate:
        angle1 = np.random.rand() * 180
        sample = rotate(sample, angle1, axes=(2, 3), reshape=False)

    if ifcontrast:
        factor = np.random.rand() * 2
        new_sample = []
        for i in range(sample.shape[1]):
            image_pil = Image.fromarray(sample[0, i])
            enhancer = ImageEnhance.Contrast(image_pil)
            image_enhanced = enhancer.enhance(factor)
            new_sample.append(np.array(image_enhanced))
        sample = np.expand_dims(new_sample, 0)

    if ifbright:
        factor = np.random.rand() * 2
        new_sample = []
        for i in range(sample.shape[1]):
            image_pil = Image.fromarray(sample[0, i])
            enhancer = ImageEnhance.Brightness(image_pil)
            image_enhanced = enhancer.enhance(factor)
            new_sample.append(np.array(image_enhanced))
        sample = np.expand_dims(new_sample, 0)

    if ifsharp:
        factor = np.random.rand() * 2
        new_sample = []
        for i in range(sample.shape[1]):
            image_pil = Image.fromarray(sample[0, i])
            enhancer = ImageEnhance.Sharpness(image_pil)
            image_enhanced = enhancer.enhance(factor)
            new_sample.append(np.array(image_enhanced))
        sample = np.expand_dims(new_sample, 0)

    if ifswap:
        if sample.shape[1] == sample.shape[2] and sample.shape[1] == sample.shape[3]:
            axisorder = np.random.permutation(3)
            sample = np.transpose(sample, np.concatenate([[0], axisorder + 1]))


    if ifflip:
        #         flipid = np.array([np.random.randint(2),np.random.randint(2),np.random.randint(2)])*2-1
        flipid = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
        sample = np.ascontiguousarray(sample[:, ::flipid[0], ::flipid[1], ::flipid[2]])

    return sample

import numpy as np
def min_max_img(img):
    min_b, max_b = 10, -100
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if(img[i,j] < min_b): min_b = img[i,j]
            if(img[i,j] > max_b): max_b = img[i,j]
    return (min_b, max_b)
def normalize_img(img):
    min_b, max_b = min_max_img(img)
    range_b = max_b - min_b
    n_img = np.zeros_like(img, dtype=float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            n_img[i, j] = (img[i, j] - min_b)/ range_b
    return n_img
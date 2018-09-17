import matplotlib.pyplot as plt #matplotlib para graficar
from numba import jit #easy paralleization
import numpy as np

import skimage.morphology as morpho
from astropy.io import fits
import skimage as sk

from scipy.stats import multivariate_normal

import glob

def gauss_elem(sz, var = 1):
  x_i, x_j = sz
  center = (x_i, x_j)
  variances = np.mat('{0},0; 0 {1}'.format(x_i * var, x_j * var))
  gauss = multivariate_normal(center, variances)
  img_gauss = np.zeros((2*x_i + 1, 2*x_j + 1), dtype=np.float)
  for i in range(2*x_i + 1):
    for j in range(2*x_j + 1):
      img_gauss[i, j] =  gauss.pdf((i,j))
  return img_gauss

def min_max_img(img):
  min_b, max_b = 10, -100
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      if(img[i,j] < min_b): min_b = img[i,j]
      if(img[i,j] > max_b): max_b = img[i,j]
  return (min_b, max_b)

def normalize_img(img, as_int=False, max_val=255):
  min_b, max_b = min_max_img(img)
  range_b = max_b - min_b
  n_img = np.zeros_like(img, dtype=float)
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      n_img[i, j] = (img[i, j] - min_b)/ range_b
  if(as_int): n_img = sk.img_as_int(n_img * max_val)
  return n_img

def readNormalizedImg(img):
  f = fits.open(img)[0].data
  while (len(f.shape) > 2): f = f[0]
  return normalize_img(f)

def readNormalizedImgMix(img1, img2):
  f1 = fits.open(img1)[0].data
  f2 = fits.open(img2)[0].data
  while (len(f1.shape) > 2): f1 = f1[0]
  while (len(f2.shape) > 2): f2 = f2[0]
  scale_fac = (f1.shape[0]/f2.shape[0] , f1.shape[1]/f2.shape[1])
  f2r = sk.transform.rescale(f2, scale_fac, mode='reflect')
  f3 = f1 + f2r
  return normalize_img(f3)

def img_area(img):
    area = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            area += img[i,j]
    return area

def get_open_close_info(im, sz):
  a = np.zeros((2 * sz))
  for i in range(sz):
    im_c = morpho.opening(im, morpho.disk(i + 1))
    a[i] = img_area(im_c)
    im_c = morpho.closing(im, morpho.disk(i + 1))
    a[i + sz] = img_area(im_c)
  return a

def get_file_list(directory, ext):
  return sorted([f for f in glob.glob('{0}/*.{1}'.format(directory, ext))])

def transpose_mtx(mtx):
    return [[mtx[i][j] for i in range(len(mtx))] for j in range(len(mtx[0]))]


def readImagesFromDirs(dirs):
  files = transpose_mtx([get_file_list(dirs[i], 'fit') for i in range(len(dirs))])
  images = []
  names = []
  for im in files:
      try:
          #could easily optimize reading process...
          img = readNormalizedImgMix(im[0], im[1])
          images.append([readNormalizedImg(im[0]), readNormalizedImg(im[1]), img])
          names.append(im)
      except:
          print('error')
  files = names
  return files, np.array(images)
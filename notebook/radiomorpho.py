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
  f1 = readNormalizedImg(img1)
  f2 = readNormalizedImg(img2)
  scale_fac = (f1.shape[0]/f2.shape[0] , f1.shape[1]/f2.shape[1])
  f2r = sk.transform.rescale(f2, scale_fac, mode='reflect', multichannel=False, anti_aliasing=True)
  f3 = f1 + f2r
  return normalize_img(f3)

def img_area(img):
    area = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            area += img[i,j]
    return area

def get_open_close_info(im, sz):
  o_area = img_area(im)
  a = np.zeros((2 * sz))
  for i in range(sz):
    im_c = morpho.opening(im, morpho.disk(i + 1))
    a[i] = (o_area - img_area(im_c)) / o_area
    im_c = morpho.closing(im, morpho.disk(i + 1))
    a[i + sz] = (o_area - img_area(im_c)) / o_area
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
          print('error reading image {0} or {1}'.format(im[0], im[1]))
  files = names
  return files, np.array(images)


# thresholding code
def threshold_kittler(img, e = 10E-6, m_iter=100):
  def new_hist_T(p, a, v):
    #-b +- sqrt( pow(b, 2) - 4ac)/ 2
    _a = (1/v[0] - 1/v[1])
    _b = -2 * (a[0]/v[0] - a[1]/v[1])
    _c = a[0]**2 / v[0] - a[1]**2 / v[1]
    _c += 2 * (np.log(np.sqrt(v[0])) - np.log(np.sqrt(v[1])))
    _c -= 2 * (np.log(p[0]) - np.log(p[1]))

    pos = ( -_b + np.sqrt(_b**2 - 4*_a*_c))/(2*_a)
    return pos

  def get_params(img, T, mask=None):
    mask = img > T
    mask_i = img < T
    p = (float(img[mask_i].shape[0])/(img.shape[0] * img.shape[1]),
         float(img[mask].shape[0])/(img.shape[0] * img.shape[1]))
    a = (img[mask_i].mean(), img[mask].mean())
    v = (img[mask_i].var(), img[mask].var())
    return(p,a,v)

  T = img.mean()
  new_t = 0
  while( m_iter ):
    m_iter -= 1
    new_t = new_hist_T(*get_params(img, T))
    if abs(new_t - T) < e: break
    else: T = new_t
  return new_t

def Kittler(im, n=8):
    """
    The reimplementation of Kittler-Illingworth Thresholding algorithm by Bob Pepin
    Works on n-bit images only
    Original Matlab code: https://www.mathworks.com/matlabcentral/fileexchange/45685-kittler-illingworth-thresholding
    Paper: Kittler, J. & Illingworth, J. Minimum error thresholding. Pattern Recognit. 19, 41–47 (1986).
    """
    bits=2**n
    h,g = np.histogram(im.ravel(),bits,[0,bits])
    h = h.astype(np.float)
    g = g.astype(np.float)
    g = g[:-1]
    c = np.cumsum(h)
    m = np.cumsum(h * g)
    s = np.cumsum(h * g**2)
    sigma_f = np.sqrt(s/c - (m/c)**2)
    cb = c[-1] - c
    mb = m[-1] - m
    sb = s[-1] - s
    sigma_b = np.sqrt(sb/cb - (mb/cb)**2)
    p =  c / c[-1]
    v = p * np.log(sigma_f) + (1-p)*np.log(sigma_b) - p*np.log(p) - (1-p)*np.log(1-p)
    v[~np.isfinite(v)] = np.inf
    idx = np.argmin(v)
    t = g[idx]
    return t

def Kittler_flot(im, n=16):
  return Kittler(im*(2**n), n) / (2**n)

def remove_data_bellow_threshold(img, threshold_alg=Kittler_flot, ratio=40):
  disk_sz = 1 + int(img.shape[0]/ratio)
  mask = img > threshold_alg(img)
  # mask = sk.morphology.opening(mask, sk.morphology.disk(disk_sz))
  img_o = np.zeros_like(img)
  for i in range(img.shape[0]):
      for j in range(img.shape[1]):
          img_o[i, j] = img[i, j] if(mask[i, j]) else 0
  return img_o

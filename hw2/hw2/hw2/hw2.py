import cv2
import numpy as np
from scipy.stats import pearsonr
from skimage import util as util
from skimage import filters as filt
from skimage.morphology import disk, square
from skimage import restoration
from scipy import ndimage
import matplotlib.pyplot as plt

def img_to_uint8(img):
    return np.uint8(img)

def show_images(images_infos):
    for name in images_infos:
        cv2.imshow(name, images_infos[name])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_correlation_coeff(image1, image2):
    coeff, _ = pearsonr(img_to_uint8(image1).flatten(), img_to_uint8(image2).flatten())
    return coeff

img = cv2.imread('./image2.jpg', 0)
img_float32 = np.float32(img)

dct_img = cv2.dct(img_float32)
idct_img = cv2.idct(dct_img)

dct_corr_coeff = get_correlation_coeff(img_float32, idct_img)

show_images({'Source image' : img, 'DCT': img_to_uint8(dct_img), 'Inverse DCT': img_to_uint8(idct_img)})

f_img = np.fft.fft2(img_float32)
if_img = np.fft.ifft2(f_img)

f_corr_coeff = get_correlation_coeff(img_float32, if_img)

show_images({'Source image' : img, 'FFT': img_to_uint8(f_img), 'Inverse FFT': img_to_uint8(if_img)})

def show_corr_coeffs(corr_coeffs):
    for coeff in corr_coeffs:
        print(f'{coeff}: {corr_coeffs[coeff]}')

    mx_coeff = max(corr_coeffs, key=corr_coeffs.get)
    print(f'Наиболее точный результат: {mx_coeff}')


corr_coeffs = {'Косинусное преобразование': dct_corr_coeff, 'Преобразование Фурье': f_corr_coeff}
show_corr_coeffs(corr_coeffs)

# print(f'Коэфф. корреляции косинусного преобразования: {dct_corr_coeff}\nКоэфф. корреляции преобразования Фурье: {f_corr_coeff}')
# if dct_corr_coeff > f_corr_coeff:
#     print ('Результат косинусного преобразования более точен!')
# else:
#     print ('Результат преобразования Фурье более точен!')

gaussian_img = util.random_noise(img, mode='gaussian', var=0.01)
salt_and_pepper_img = util.random_noise(img, mode='s&p', amount=0.3)
speckle_img = util.random_noise(img, mode='speckle',var=0.01)

mask_size = 3

def get_psf(ds):
    psf = ndimage.gaussian_filter(np.ones((int(ds), int(ds))), sigma=1.1)
    psf = psf / psf.sum() 
    return psf

balance = 0.05

median_img = filt.median(gaussian_img, footprint=disk(mask_size))
rank_img = filt.rank.median(salt_and_pepper_img, footprint=disk(mask_size))
wiener_img = restoration.wiener(speckle_img, get_psf(mask_size), balance=balance)

show_images({'Source image' : img, 'Gaussian noise': gaussian_img, 'S&P noise': salt_and_pepper_img, 'Speckle noise': speckle_img})
show_images({'Source image' : img, 'Median filter': median_img, 'Rank filter': rank_img, 'Wiener filter': wiener_img})

median_corr_coeff = get_correlation_coeff(img_float32, median_img)
rank_corr_coeff = get_correlation_coeff(img_float32, rank_img)
wiener_corr_coeff  = get_correlation_coeff(img_float32, wiener_img)

corr_coeffs = {'Медианный фильтр': median_corr_coeff, 'Ранговый фильтр': rank_corr_coeff, 'Фильтр Винера': wiener_corr_coeff}
show_corr_coeffs(corr_coeffs)

masks = [1, 3, 7]
for mask in masks:
    print(f"Фильтрация, размер маски = {mask}")
    median_img = filt.median(gaussian_img, footprint=disk(mask))
    rank_img = filt.rank.median(salt_and_pepper_img, footprint=disk(mask))
    wiener_img = restoration.wiener(speckle_img, get_psf(mask), balance=balance)
    show_images({'Source image' : img, f'Median filter (mask = {mask})': median_img, f'Rank filter (mask = {mask})': rank_img, f'Wiener filter (mask = {mask})': wiener_img})


mask_sizes = np.arange(1, 19, 1)

median_coeffs = [get_correlation_coeff(img_float32, filt.median(gaussian_img, footprint=disk(i))) for i in mask_sizes]
rank_coeffs = [get_correlation_coeff(img_float32, filt.rank.median(salt_and_pepper_img, footprint=disk(i))) for i in mask_sizes]
wiener_coeffs = [get_correlation_coeff(img_float32, restoration.wiener(speckle_img, get_psf(i), balance=balance)) for i in mask_sizes]

plt.figure()
plt.subplot(2,2,1)
plt.plot(mask_sizes, median_coeffs, 'o-', label='Медианная фильтрация')
plt.xlabel('Размер маски')
plt.ylabel('Коэффициент корреляции')
plt.grid(True, alpha=0.5)
plt.ylim(-1.0, 1.1)
plt.axhline(y=0.0, color='black', linestyle='--')
plt.legend()
plt.title('Медианная фильтрация')

plt.subplot(2,2,2)
plt.plot(mask_sizes, rank_coeffs, 'o-', label='Ранговая фильтрация')
plt.xlabel('Размер маски')
plt.ylabel('Коэффициент корреляции')
plt.grid(True, alpha=0.5)
plt.ylim(-1.0, 1.1)
plt.axhline(y=0.0, color='black', linestyle='--')
plt.legend()
plt.title('Ранговая фильтрация')

plt.subplot(2,2,3)
plt.plot(mask_sizes, wiener_coeffs, 'o-', label='Фильтрация Винера')
plt.xlabel('Размер маски')
plt.ylabel('Коэффициент корреляции')
plt.grid(True, alpha=0.5)
plt.ylim(-1.0, 1.1)
plt.axhline(y=0.0, color='black', linestyle='--')
plt.legend()
plt.title('Фильтрация Винера')

plt.show()

f_img = 10 * np.log(np.abs(np.fft.fftshift(f_img)) + 1)
rows, cols = f_img.shape

center_x, center_y = cols // 2, rows // 2
y, x = np.indices((rows, cols))
r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
r = r.astype(int)
    
tbin = np.bincount(r.ravel(), f_img.ravel())
nr = np.bincount(r.ravel())
radial_profile = tbin / nr
    
plt.plot(radial_profile[:min(rows, cols)//2])
plt.title('Зависимость спектра яркости от частоты')
plt.xlabel('Радиус (пиксели)')
plt.ylabel('Амплитуда')
plt.grid(True, alpha=0.5)
    
plt.show()

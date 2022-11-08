import math
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

get_image_dimensions = lambda img: (*img.shape, 1) if len(img.shape) == 2 else img.shape

def fix_image_colors(image):
    w, h, depth = get_image_dimensions(image)
    if depth == 1:
         return image
    b, g, r = cv.split(image)
    return cv.merge([r, g, b]) 

def imshow(src, cmap = 'gray', vmin=0, vmax=255):
    plt.axis('off')
    plt.imshow(src, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.show()

def imshow2(*images, titles=None, cmap='gray', vmin=0, vmax=255, cols=3, titleColor = "black"):
    rows = math.ceil(len(images) / cols)
    plt.figure(figsize=(16, 8))
    plt.axis('off')
    for i in range(len(images)):
        _, _, depth = get_image_dimensions(images[i])
        plt.subplot(rows, cols, i+1)
        if depth == 1:
            plt.imshow(images[i], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            print()
            plt.imshow(fix_image_colors(images[i]))
        if titles is not None:
            plt.title(titles[i], color = titleColor)
        plt.xticks([])
        plt.yticks([])
    plt.show()

def pad_image(image: np.ndarray, size = 1) -> np.ndarray:
    img_height, img_width = image.shape
    img_expanded = np.zeros((size * 2 +img_height, size * 2 + img_width), np.float32)
    img_expanded[size:size + img_height, size:size + img_width] = image
    return img_expanded

def create_gaussian_kernel(size: int = 3, sig: int = 1) -> np.ndarray:
    if size % 2 == 0:
        raise ValueError('Kernel size must be odd')
    ax = np.linspace(- size // 2, size // 2, size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def create_kernel(size: int = 3, value: any = None) -> np.ndarray:  # type: ignore
    if size % 2 == 0:
        raise ValueError('Kernel size must be odd')

    if value is None:
        kernel = np.zeros((size, size), np.float32)
        kernel[size // 2][size // 2] = 1
        return kernel

    if isinstance(value,(list, tuple, np.ndarray)):
        tmp_value = np.array(value)
        if tmp_value.ndim == 1:
            tmp_value = tmp_value.reshape((size, size))
        
        rows, cols = tmp_value.shape
        if rows != cols:
            raise ValueError('Kernel must be squared')

        if rows % 2 == 0:
            raise ValueError('Kernel size must be odd')
        
        return tmp_value
        
    return np.ones((size, size), np.float32) * value

def transform(image: np.ndarray, kernel: np.ndarray = np.ones((3, 3), np.float32)) -> np.ndarray:
    kern_height, kern_width = kernel.shape
    if kern_height % 2 == 0 or kern_width % 2 == 0:
        raise ValueError('Kernel size must be odd')
    
    extra_cols_rows = kern_width // 2
    img_height, img_width = image.shape
    img_expanded = pad_image(image, extra_cols_rows)
    img_modified = image.copy().astype('float')

    for i in range(img_height):
        for j in range(img_width):
            part = img_expanded[i:i + kern_height, j:j + kern_width]
            img_modified[i, j] = np.sum(part * kernel)

    return img_modified.astype('int')

def median_transform(image: np.ndarray, kern_size:int = 3 ) -> np.ndarray:
    if kern_size % 2 == 0:
        raise ValueError('Kernel size must be odd')
    
    extra_cols_rows = kern_size // 2
    img_height, img_width = image.shape
    img_expanded = pad_image(image, extra_cols_rows)
    img_modified = image.copy().astype('float')

    for i in range(img_height):
        for j in range(img_width):
            part = img_expanded[i:i + kern_size, j:j + kern_size]
            img_modified[i, j] = np.median(part)

    return img_modified.astype('int')

def gaussian_transform(image: np.ndarray, kernel: np.ndarray = np.ones((3, 3), np.float32)) -> np.ndarray:
    kern_height, kern_width = kernel.shape
    if kern_height % 2 == 0 or kern_width % 2 == 0:
        raise ValueError('Kernel size must be odd')
    
    extra_cols_rows = kern_width // 2
    img_height, img_width = image.shape
    img_expanded = pad_image(image, extra_cols_rows)
    img_modified = image.copy().astype('float')

    for i in range(img_height):
        for j in range(img_width):
            part = img_expanded[i:i + kern_height, j:j + kern_width]
            img_modified[i, j] = part[i, j] + kernel

    return img_modified.astype('int')
    
def generate_histogram(img: np.ndarray) -> list:
    return [(img == x).sum() for x in range(0, 256)]

def create_laplacian_kernel(alpha: float) -> np.ndarray:
    return np.array([[alpha, 1-alpha, alpha], [1 - alpha, -4, 1 - alpha], [alpha, 1-alpha, alpha]]) * 1 / (1 + alpha)

def create_log_kernel(size: int, alpha: float) -> np.ndarray:
    LoG  = lambda x, y, alpha: (-1 / np.pi * alpha) * (1 - (x**2 + y**2) / 2 * alpha **2) * np.exp(-(x**2 + y**2) / 2 * alpha **2)
    return np.array([LoG(x, y, alpha) for x in range(size) for y in range(size)]).reshape((size, size))

def otsu(img: np.ndarray) -> tuple:
    hist = generate_histogram(img)
    best_threshold = 0
    var_intraclass = math.inf    
    p = hist / np.sum(hist)
    for t in range(1, 256):
        g1 = p[0:t]
        g2 = p[t:255]
        i1 = np.arange(t)
        i2 = np.arange(t, 255)
        q1 = np.sum(g1)
        q2 = np.sum(g2)
        prom1 = np.sum(i1 * g1 / q1)
        prom2 = np.sum(i2 * g2 / q2)
        var1 = np.sum(((i1 - prom1)**2) * g1 / q1)
        var2 = np.sum(((i2 - prom2)**2) * g2 / q2)
        var_ic = q1 * var1 + q2 * var2
        if var_ic < var_intraclass:
            var_intraclass = var_ic
            best_threshold = t

    return best_threshold, var_intraclass
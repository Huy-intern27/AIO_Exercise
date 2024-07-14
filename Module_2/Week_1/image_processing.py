import numpy as np  # type: ignore
import matplotlib.image as mpimg  # type: ignore
import matplotlib.pyplot as plt  # type: ignore


def light_ness(img):
    max_img = np.max(img[..., :3], axis=2)
    min_img = np.min(img[..., :3], axis=2)
    gray_img = max_img * 0.5 + min_img * 0.5
    return gray_img


def average(img):
    gray_img = np.mean(img, axis=2)
    return gray_img


def luminosity(img):
    gray_img = 0.21 * img[..., 0] + 0.72 * img[..., 1] + 0.07 * img[..., 2]
    return gray_img


img = mpimg.imread('Module_2/Week_1/data/dog.jpeg')
gray_img = luminosity(img)
print(gray_img[0, 0])

plt.imshow(gray_img, cmap='gray')
plt.axis('off')
plt.show()

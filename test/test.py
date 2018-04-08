import numpy as np
print("Imported numpy:\n" + np.__version__)


import matplotlib as mpl
print("Imported matplotlib:\n" + mpl.__version__)
import matplotlib.pyplot as plt


import cv2
print("Imported cv2:\n" + cv2.__version__)

img = cv2.imread('lena.jpg')

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(121)
plt.imshow(img_gray, cmap='gray')

sift = cv2.xfeatures2d.SIFT_create()
print("SIFT imported in package")
kp, desc = sift.detectAndCompute(img_gray, None)
plt.subplot(122)
plt.imshow(cv2.drawKeypoints(img_gray, kp, img.copy()))

plt.show()

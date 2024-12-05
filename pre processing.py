#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[10]:


img1 = cv2.imread(r"C:\Users\Mohammad\Desktop\29030.jpg")

img_yuv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)


img_yuv1[:,:,0] = cv2.equalizeHist(img_yuv1[:,:,0])


img_output = cv2.cvtColor(img_yuv1, cv2.COLOR_YUV2BGR)

cv2.imshow('Color input image', img1)
cv2.imshow('Histogram equalized', img_output)

plt.hist(img_yuv1[:,:,0])
plt.show()

cv2.waitKey(0)


# In[10]:


image = cv2.imread(r"C:\Users\Mohammad\Desktop\29030.jpg")


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


channels = cv2.split(image_rgb)
colors = ('r', 'g', 'b')
channel_ids = (0, 1, 2)


plt.figure()
plt.title('Car')
plt.xlabel('Bins')
plt.ylabel('Number of Pixels')

for channel_id, color in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        channels[channel_id], bins=256, range=(0, 255)
    )

    plt.plot(bin_edges[0:-1], histogram, color=color)

plt.xlim([0, 255])
plt.show()


# In[8]:


img2 = cv2.imread(r"C:\Users\Mohammad\Desktop\124084.jpg")

img_yuv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YUV)


img_yuv2[:,:,0] = cv2.equalizeHist(img_yuv2[:,:,0])


img_output = cv2.cvtColor(img_yuv2, cv2.COLOR_YUV2BGR)

cv2.imshow('Color input image', img2)
cv2.imshow('Histogram equalized', img_output)

plt.hist(img_yuv2[:,:,0])
plt.show()


cv2.waitKey(0)


# In[11]:


image = cv2.imread(r"C:\Users\Mohammad\Desktop\124084.jpg")


image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


channels = cv2.split(image_rgb)
colors = ('r', 'g', 'b')
channel_ids = (0, 1, 2)


plt.figure()
plt.title('Flower')
plt.xlabel('Bins')
plt.ylabel('Number of Pixels')

for channel_id, color in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        channels[channel_id], bins=256, range=(0, 255)
    )

    plt.plot(bin_edges[0:-1], histogram, color=color)

plt.xlim([0, 255])
plt.show()


# In[2]:


img3 = cv2.imread(r"C:\Users\Mohammad\Desktop\299091.jpg")

img_yuv3= cv2.cvtColor(img3, cv2.COLOR_BGR2YUV)


img_yuv3[:,:,0] = cv2.equalizeHist(img_yuv3[:,:,0])


img_output = cv2.cvtColor(img_yuv3, cv2.COLOR_YUV2BGR)

cv2.imshow('Color input image', img3)
cv2.imshow('Histogram equalized', img_output)

plt.hist(img_yuv3[:,:,0])
plt.show()

cv2.waitKey(0)


# In[8]:


image = cv2.imread(r"C:\Users\Mohammad\Desktop\299091.jpg")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


channels = cv2.split(image_rgb)
colors = ('r', 'g', 'b')
channel_ids = (0, 1, 2)


plt.figure()
plt.title('Pyramid')
plt.xlabel('Bins')
plt.ylabel('Number of Pixels')

for channel_id, color in zip(channel_ids, colors):
    histogram, bin_edges = np.histogram(
        channels[channel_id], bins=256, range=(0, 255)
    )

    plt.plot(bin_edges[0:-1], histogram, color=color)

plt.xlim([0, 255])
plt.show()


# In[12]:


c = 255 / np.log(1 + np.max(img1)) 
log_image = c * (np.log(img1 + 1)) 
   

log_image = np.array(log_image, dtype = np.uint8) 
   
 
cv2.imshow('log transformed', log_image)
cv2.waitKey(0)


# In[9]:


c = 255 / np.log(1 + np.max(img2)) 
log_image = c * (np.log(img2 + 1)) 
   

log_image = np.array(log_image, dtype = np.uint8) 
   
 
cv2.imshow('log transformed', log_image)
cv2.waitKey(0)


# In[4]:


c = 255 / np.log(1 + np.max(img3)) 
log_image = c * (np.log(img3 + 1)) 
   

log_image = np.array(log_image, dtype = np.uint8) 
   
 
cv2.imshow('log transformed', log_image)
cv2.waitKey(0)


# In[9]:


import cv2


image = cv2.imread(r"C:\Users\Mohammad\Desktop\299091.jpg")


scale = 1.5  


height, width = image.shape[:2]


new_height, new_width = int(height * scale), int(width * scale)


zoomed_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


cv2.imshow('Original Image of pyramid', image)
cv2.imshow('Zoomed Image of pyramid', zoomed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[8]:


import cv2


image = cv2.imread(r"C:\Users\Mohammad\Desktop\124084.jpg")


scale = 1.5  


height, width = image.shape[:2]


new_height, new_width = int(height * scale), int(width * scale)


zoomed_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


cv2.imshow('Original Image of flower', image)
cv2.imshow('Zoomed Image of flower', zoomed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[3]:


import cv2


image = cv2.imread(r"C:\Users\Mohammad\Desktop\29030.jpg")


scale = 1.5  


height, width = image.shape[:2]


new_height, new_width = int(height * scale), int(width * scale)


zoomed_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


cv2.imshow('Original Image of car', image)
cv2.imshow('Zoomed Image of car', zoomed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[2]:


import cv2
import numpy as np


image = cv2.imread(r"C:\Users\Mohammad\Desktop\393035.jpg")

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


pixel_values = image_rgb.reshape((-1, 3))
pixel_values = np.float32(pixel_values)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
k = 3  
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


centers = np.uint8(centers)

segmented_image = centers[labels.flatten()]


segmented_image = segmented_image.reshape(image_rgb.shape)


gray_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)


edges = cv2.Canny(gray_image, 100, 200)


_, thresholded = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)


cv2.imshow('Original Image', image_rgb)
cv2.imshow('Segmented Image', segmented_image)
cv2.imshow('Edge Detection', edges)
cv2.imshow('Thresholded Image', thresholded)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


import cv2
import numpy as np
from skimage.feature import greycomatrix, greycoprops, corner_harris, corner_peaks
from scipy.fftpack import fft2, fftshift
from skimage.measure import moments_hu
import matplotlib.pyplot as plt


image = cv2.imread(r"C:\Users\Mohammad\Desktop\87015.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


edges = cv2.Canny(gray_image, 100, 200)
image_edges = image.copy()
image_edges[edges != 0] = [0, 255, 0]  


contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_contours = cv2.drawContours(image.copy(), contours, -1, (0, 0, 255), 2) 


mask = np.zeros(image.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (50, 50, image.shape[1]-50, image.shape[0]-50)
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
segmented_image = image * mask2[:, :, np.newaxis]


sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray_image, None)
image_keypoints = cv2.drawKeypoints(image.copy(), keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


colors = ('b', 'g', 'r')
plt.figure()
plt.title('Color Histogram')
plt.xlabel('Bins')
plt.ylabel('Number of Pixels')
for i, color in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])


cv2.imshow('Edges', image_edges)
cv2.imshow('Contours', image_contours)
cv2.imshow('Keypoints', image_keypoints)
cv2.imshow('Segmented Image', segmented_image)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:


import cv2
import numpy as np


image = cv2.imread(r"C:\Users\Mohammad\Desktop\124084.jpg")
mask = np.zeros(image.shape[:2], np.uint8)


bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)


rect = (50, 50, image.shape[1]-50, image.shape[0]-50)  


cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')


segmented_image = image * mask2[:, :, np.newaxis]


cv2.imshow('Original Image', image)
cv2.imshow('Background Removed', segmented_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


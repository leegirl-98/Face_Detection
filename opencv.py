#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2 as cv


# In[2]:


img = cv.imread(r'C:\Users\Welcome\Desktop\fav pic\100MEDIA\IMAG1756.jpg')


# In[3]:


cv.imshow("itsme", img)
cv.waitKey(0)


# In[4]:


resized=cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow("resized",resized)
cv.waitKey(0)


# # converting to gray scale

# In[5]:


gray=cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)
cv.waitKey(0)


# # BLUR

# In[6]:


blur=cv.GaussianBlur(resized,(7,7),cv.BORDER_DEFAULT)
cv.imshow("blur",blur)
cv.waitKey(0)


# # edge casecade
# 

# In[7]:


canny=cv.Canny(blur,125,175)
cv.imshow("Canny edge",canny)
cv.waitKey(0)


# # Dilating the image

# In[8]:


dilating=cv.dilate(canny,(7,7),iterations=3)
cv.imshow("Dilate",dilating)
cv.waitKey(0)


# # Eroding

# In[22]:


eroded=cv.erode(dilating,(7,7),iterations=3)
cv.imshow("eroded",eroded)
cv.waitKey(0)


# # Cropping

# In[41]:


cropping=img[50:200,3:450]
cv.imshow("cropping",cropping)
cv.waitKey(0)


# # contours

# In[10]:


import cv2 as cv
import numpy as np


# # blank

# In[11]:


blank=np.zeros(resized.shape,dtype='uint8')
cv.imshow("Blank",blank)
cv.waitKey(0)


# In[12]:


gray=cv.cvtColor(resized,cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)
cv.waitKey(0)


# In[13]:


blur=cv.GaussianBlur(resized,(5,5),cv.BORDER_DEFAULT)
cv.imshow("blur",blur)
cv.waitKey(0)


# In[11]:


canny=cv.Canny(blur,125,175)
cv.imshow("canny",canny)
cv.waitKey(0)


# In[19]:


ret,thresh=cv.threshold(gray,125,255,cv.THRESH_BINARY)
cv.imshow("thresh",thresh)
cv.waitKey(0)


# In[13]:


contours,hierarchies=cv.findContours(canny,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found!')


# In[15]:


cv.drawContours(blank,contours,-1,(0,0,255),1)
cv.imshow("contour drawn",blank)
cv.waitKey(0)


# In[15]:


threshold,thresh_inv=cv.threshold(gray,150,255,cv.THRESH_BINARY_INV)
cv.imshow("thresh",thresh_inv)
cv.waitKey(0)


# In[25]:


adaptive_thresh=cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,9)
cv.imshow("adaptive threshold",adaptive_thresh)
cv.waitKey(0)


# In[14]:


contours,hierarchies=cv.findContours(thresh_inv,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
print(f'{len(contours)} contour(s) found' )


# In[ ]:





# In[ ]:





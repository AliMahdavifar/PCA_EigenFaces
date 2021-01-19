# !pip install wget

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import wget
import math
import matplotlib.animation as animation
import eigenfaces as eg

[X, y] = eg.readImages()

# create a mean face from all dataset faces
XMat = eg.asRowMatrix(X)
meanImage = np.reshape(XMat.mean(axis=0), X[0].shape)
plt.imshow(meanImage, cmap=plt.cm.gray)
plt.title('Mean Face')
plt.show()

eigenValues, eigenVectors, mean = eg.pca(XMat, y)

IMAGE_IDX = 10 # image idx in dataset

# actual image
plt.imshow(X[IMAGE_IDX], cmap=plt.cm.gray)
plt.show()

# create reconstructed images
COUNT = 6 # count of first eigenVectors used to reconstruct the image
reconImages = []
for numEvs in range (1, COUNT+1):
    P = eg.project(eigenVectors[:, 0:numEvs], X[IMAGE_IDX].reshape(1, -1), mean)
    R = eg.reconstruct(eigenVectors[:, 0:numEvs], P, mean)
    reconImages.append(R.reshape(X[0].shape))

# plot reconstructed images
ROWS = math.ceil(COUNT/3)
fig = plt.figure(figsize=(12, ROWS * 4))
for i in range(0, COUNT):
    plt.subplot(ROWS, 3, i+1)
    plt.imshow(reconImages[i], cmap = plt.cm.gray)
    plt.title('#{}'.format(i+1))

# create reconstructed images
numEvsSet = [1, 10 , 100, 200, 400, 500, 535] # these no. of eigenVectors will be used to reconstruct the image.
COUNT = len(numEvsSet)
reconImages = []
for numEvs in numEvsSet:
    P = eg.project(eigenVectors[:, 0:numEvs], X[IMAGE_IDX].reshape(1, -1), mean)
    R = eg.reconstruct(eigenVectors[:, 0:numEvs], P, mean)
    reconImages.append(R.reshape(X[0].shape))

# plot reconstructed images
ROWS = math.ceil(COUNT/3)
fig = plt.figure(figsize=(12, ROWS * 4))
for i in range(0, COUNT):
    plt.subplot(ROWS, 3, i+1)
    plt.imshow(reconImages[i], cmap = plt.cm.gray)
    plt.title("Reconstruction:"+ str(numEvsSet[i]) + " Components" )

# create an animation of reconstruction
fig = plt.figure()

ims = []
reconImages = []

for numEvs in range (0 , eigenVectors.shape[1]):
    if numEvs % 50 ==0:
      print ("Progress: %.2f " % (numEvs *100/ eigenVectors.shape[1]), "%")

    title = plt.text(125.5,0.85, "", bbox={'facecolor':'w', 'alpha':1, 'pad':5},
                 ha="center")

    P = eg.project(eigenVectors[:, 0:numEvs], X[IMAGE_IDX].reshape(1, -1), mean)
    R = eg.reconstruct(eigenVectors[:, 0:numEvs], P, mean)
    reconImages=(R.reshape(X[0].shape))
    title.set_text("Reconstruction:"+ str(numEvs) + " Components")
    im = plt.imshow(reconImages, cmap = plt.cm.gray)
    ims.append([im, title])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)

ani.save('dynamic_images.mp4')

plt.show()
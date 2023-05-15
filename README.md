# Fashion-MNIST-image-classifier

## Introduction
The goal of this project is to develop a machine learning model to classify between images of T-shirts and dress-shirts.

## Dataset
The dataset consists of 12000 flattened 28x28 pixel gray-scale images, which are stored in TrainData.csv. The true labels for the examples are provided in TrainLabels.csv, where T-shirts are labeled with 1 and dress-shirts with -1. TestData.csv contains the test examples.

## Usage
To load the data, use the following code:
```
Xtr=np.loadtxt("TrainData.csv")
Ytr=np.loadtxt("TrainLabels.csv")
Xts=np.loadtxt("TestData.csv")
```

To visualize a training example at index 10, use the following code:
```
import matplotlib.pyplot as plt
plt.imshow(Xtr[10].reshape([28,28]))
```

## Feature Engineering
For this assignment, we need to extract the HOG (Histogram of Oriented Gradients) features from the images. A tutorial on how to do this can be found at: https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/.

## Output
As part of this project, an image that has my name and roll number written in it under a car picture has been included. The extracted HOG features of this image are included in the output folder.

## Plotting
To plot any one image from the dataset and its extracted features, we can use the following code:
```
import matplotlib.pyplot as plt
from skimage.feature import hog

image = Xtr[0].reshape([28,28])
fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
                    cells_per_block=(1, 1), visualize=True, multichannel=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.imshow(image, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
ax2.set_title('HOG features')
plt.show()
```

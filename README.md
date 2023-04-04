# fashion-MNIST-image-classifier

The goal in this assignment is to develop a machine learning model that can classify between images of T-shirts and dress-shirts. You are given the following files:
• TrainData.csv: It contains 12000 training examples. Each row contains 784 values. The dataset has been derived from Fashion-MNIST dataset. Each example is a flattened 28x28 pixel gray-scale image. You can reshape the examples to visualize what each image looks like.

• TrainLabels.csv: This file contains true labels for the examples in TrainExamples.csv
(T-shirts = 1 , dress-shirts = -1)

• TestData.csv: This file contains test examples.

• You can load train and test data using the following code:
Xtr=np.loadtxt("TrainData.csv")
Ytr=np.loadtxt("TrainLabels.csv")
Xts=np.loadtxt("TestData.csv")

• To visualize an example (say trainining example at index 10, you can use the following
code):
import matplotlib.pyplot as plt
plt.imshow(Xtr[10].reshape([28,28]))

## Requirements:
1. You are required to extract hog features from these images for the classification help
at: https://www.analyticsvidhya.com/blog/2019/09/feature-engineering-images-introduction-hog-feature-descriptor/

2. Output the extracted version of yourself in the output (If not comfortable create an image that has your name and roll number written in it under a car picture)

3. Now use the dataset provided and plot any one image from there like

4. Next plot the original image and its extracted features

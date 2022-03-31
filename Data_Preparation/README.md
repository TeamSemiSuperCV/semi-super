# Dataset Preparation

This project requires data to be stored in a [TensorFlow Datasets](https://www.tensorflow.org/datasets/overview) container.

Performing EDA on the original dataset revealed a wide range of aspect-ratios and raw image sizes. Since the ResNet50 model that we'll be using as the backbone network for this project has a fixed input dimension of 224 x 224 x 3, in order to minimize distortion during training and inference, we cropped images that were either extremely tall, or extremely wide. We also normalized images to a height of 800 pixels.

Here are the steps to generate the TFDS Dataset:

1. [CheckFiles-XRay.ipynb](CheckFiles-XRay.ipynb) : Perform EDA on original images to determine *thresholds for cropping* and *target image size*
2. [Resize_XRay.ipynb](Resize_XRay.ipynb) : Perform Cropping and Resizing
3. [CheckFiles-XRay-Cropped.ipynb](CheckFiles-XRay-Cropped.ipynb) : Check results of Cropping and Resizing
4. [Generate_TFDS_Dataset.ipynb](Generate_TFDS_Dataset.ipynb) : Generate TFDS Dataset

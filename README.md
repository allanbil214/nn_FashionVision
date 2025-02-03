# FashionVision: A CNN Classifier for Fashion MNIST

## Overview
FashionVision is a convolutional neural network (CNN) designed to classify images from the Fashion MNIST dataset. This project showcases the development of a high-performance model capable of achieving over 90% accuracy on both training and validation sets. The project is structured to facilitate easy understanding and replication, making it an excellent resource for those interested in image classification using deep learning.

## Project Features
- **High Accuracy**: Achieves over 90% accuracy on the Fashion MNIST dataset.
- **Clean Architecture**: Utilizes a straightforward CNN model without unnecessary complexity.
- **Efficient Training**: Optimized preprocessing and model structure for quick and effective training.

## Dataset
The **Fashion MNIST** dataset is a collection of 70,000 grayscale images in 10 categories. Each image is 28x28 pixels, representing various fashion items such as t-shirts, trousers, and sneakers.

- **Training Set**: 60,000 images
- **Test Set**: 10,000 images

You can find more about the dataset [here](https://github.com/zalandoresearch/fashion-mnist).

## Model Architecture
The model uses a series of convolutional and pooling layers, followed by dense layers to classify the images into one of the ten categories. Here's a breakdown of the architecture:

1. **Input Layer**: (28, 28, 1) for grayscale images.
2. **Convolutional Layer 1**: 32 filters, 3x3 kernel, ReLU activation.
3. **MaxPooling Layer 1**: 2x2 pool size.
4. **Convolutional Layer 2**: 64 filters, 3x3 kernel, ReLU activation.
5. **MaxPooling Layer 2**: 2x2 pool size.
6. **Convolutional Layer 3**: 64 filters, 3x3 kernel, ReLU activation.
7. **Flatten Layer**: Converts 2D matrices to 1D vectors.
8. **Dense Layer**: 128 units, ReLU activation.
9. **Dropout Layer**: 50% dropout rate to prevent overfitting.
10. **Output Layer**: 10 units (one for each class), softmax activation.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/FashionVision.git
    cd FashionVision
    ```

2. Install the required dependencies:
    ```bash
    pip install tensorflow
    ```

## Usage
Run the script to train the model:

```bash
python nn_FashionVision.py
```

The trained model will be saved as `model.h5`.

## Performance
The model is trained for 10 epochs and achieves more than 90% accuracy on both the training and validation datasets.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Dataset sourced from Zalando's [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist).
- Developed using TensorFlow and Keras.


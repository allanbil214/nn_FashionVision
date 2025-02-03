# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

def solution_model():
    fashion_mnist = tf.keras.datasets.fashion_mnist

    # Load and preprocess the Fashion MNIST data
    (training_images, training_labels), (test_images, test_labels) = fashion_mnist.load_data()
    
    # Reshape images to (28, 28, 1) and normalize pixel values to be between 0 and 1
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(10000, 28, 28, 1)
    test_images = test_images / 255.0

    # Build the model with convolutional and pooling layers
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])

    # Compile the model using the Adam optimizer and sparse categorical crossentropy loss
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Train the model with training and validation data
    model.fit(training_images, training_labels, 
              epochs=10,
              validation_data=(test_images, test_labels),
              verbose=1)
              
    return model

if __name__ == '__main__':
    # Create and train the model
    model = solution_model()

    # Save the trained model
    model.save("model.h5")

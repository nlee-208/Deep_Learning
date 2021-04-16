import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, validation_size=5000)

print(np.shape(mnist.validation.images))
print(np.shape(mnist.validation.labels))
print(np.shape(mnist.train.images))
print(np.shape(mnist.train.labels))
print(np.shape(mnist.test.images))
print(np.shape(mnist.test.labels))
print(mnist.test.labels[9999]) # 6 -> [0 0 0 0 0 0 1 0 0 0]
plt.imshow(
        mnist.test.images[9999].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
plt.show()
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Display the fourth digit
plt.imshow(train_images[4], cmap='gray')  # Index 3 corresponds to the fourth digit
plt.title(f'Label: {train_labels[4]}')  # Display the label for the digit
plt.axis('off')  # Hide axis
plt.show()
plt.savefig('dis.png')


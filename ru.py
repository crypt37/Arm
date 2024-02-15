import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_image_and_histogram(image, augmented_image):
    # Convert images to RGB format for displaying with matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    augmented_image_rgb = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)

    # Plot the original and augmented images side by side
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(augmented_image_rgb)
    axes[1].set_title('Augmented Image')
    axes[1].axis('off')

    # Calculate and plot histograms for the original and augmented images
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].hist(image_rgb[:,:,0].ravel(), bins=256, color='red', alpha=0.7)
    axes[0, 0].set_title('Red Channel Histogram (Original)')
    axes[0, 1].hist(augmented_image_rgb[:,:,0].ravel(), bins=256, color='red', alpha=0.7)
    axes[0, 1].set_title('Red Channel Histogram (Augmented)')

    axes[1, 0].hist(image_rgb[:,:,1].ravel(), bins=256, color='green', alpha=0.7)
    axes[1, 0].set_title('Green Channel Histogram (Original)')
    axes[1, 1].hist(augmented_image_rgb[:,:,1].ravel(), bins=256, color='green', alpha=0.7)
    axes[1, 1].set_title('Green Channel Histogram (Augmented)')

    for ax in axes.flatten():
        ax.set_xlim([0, 256])
        ax.set_ylim([0, 5000])  # Adjust the y-axis limit as needed

    plt.show()

if __name__ == "__main__":
    # Replace 'your_image_path.jpg' with the path to your image
    image_path = '/home/neko/papers/nlp/disp.jpg'
    image = cv2.imread(image_path)

    # Augment the image (you can use any image augmentation technique you like)
    # Here, we simply flip the image horizontally using OpenCV
    augmented_image = cv2.flip(image, 1)

    # Plot the original and augmented images along with their histograms
    plot_image_and_histogram(image, augmented_image)

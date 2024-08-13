import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys
import torchvision.transforms as transforms
from torch import Tensor

def load_and_display_mat_image(file_path):
    # Load the .mat file
    mat_data = scipy.io.loadmat(file_path)

    # Assuming the image data is stored in a variable within the .mat file, you will need to find the key
    # Here I'm using a generic key 'image_data', replace it with the correct key
    #if 'image_data' in mat_data:
    #    image_data = mat_data['image_data']
    #else:
        # If you're unsure about the key, you can print all keys
    #     print("Available keys in the .mat file:", mat_data.keys())
    #    return
    image_data=mat_data['ground_mask']
    # If the data needs to be reshaped or transformed, it can be done here.
    # Assuming image_data is in a 2D or 3D numpy array format that can be directly displayed
    if len(image_data.shape) == 2:  # Grayscale image
        plt.imshow(image_data, cmap='gray')
    elif len(image_data.shape) == 3:  # Color image
        plt.imshow(image_data)
    else:
        print("Unknown image format!")
        return

    # Display the image
    plt.axis('off')
    plt.show()


def load_image_to_tensor(file_path:str)->Tensor:
    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No Image Exists at {file_path}")
    
    # Open the image using PIL
    try:
        image = Image.open(file_path)
        #print(image)  
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        #print(image_tensor)
        #print(image_tensor.shape)
        return image_tensor
    except Exception as e:
        raise ValueError(f"Failed to Open: {e}")


def display_image_tensor(image_tensor:Tensor):
    # Display the image
    image_numpy = image_tensor.permute(1, 2, 0).numpy()
    plt.imshow(image_numpy)
    plt.axis('off')  # Hide axis
    plt.show()

#usage:
#load_and_display_mat_image('COCO_val2014_000000000074.jpg_1.mat')  
#img = load_image_to_tensor('COCO_val2014_000000000074.jpg')
#print(img)
#display_image_tensor(img)
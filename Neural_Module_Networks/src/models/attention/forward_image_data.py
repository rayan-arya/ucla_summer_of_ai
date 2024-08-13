import os
import sys
from torch import Tensor
import torch.nn as nn
# Get the directory of the current file
current_directory = os.path.dirname(os.path.abspath(__file__))

# Move up to the 'src' directory
project_directory = os.path.dirname(os.path.dirname(current_directory))
#print(project_directory)

# Set the environment variable to the 'src' directory
os.environ['PROJECT_DIR'] = project_directory

# Now you can access this environment variable elsewhere in your project
project_dir = os.getenv('PROJECT_DIR')
#print("Project Directory:", project_dir)

# Example usage: importing a module from 'src/utils/io.py'
sys.path.append(project_dir)  # Add 'src' to sys.path
from utils.io import load_image_to_tensor, display_image_tensor  # Now you can import modules from src/utils

#navigate to ucla summer of ai folder
#activate environment: source ucla_env/bin/activate
#navigate to src folder inside NMN
#usage: python3 models/attention/forward_image_data.py 
import torch
import numpy as np

def forward_image_data(image_data:Tensor, dropout:bool)->Tensor:
	batch_size = 1
	num_channels = 3
	att_hidden = 200
	batch_size, num_channels, image_height, image_width = image_data.size()
	image_size = image_height*image_width
	#self.batch_size = batch_size
	#self.num_channels = num_channels
	#self.image_size = image_height * image_width
	print("Image data: ")
	print(image_data)
	print(image_data.shape)

	#image_data = torch.randn(batch_size,num_channels,image_width,image_height)
	#print(image_data)
	#print(image_data.shape)

	image_data_rs = image_data.view(batch_size, num_channels, image_size, 1)
	print(image_data_rs)
	print(image_data_rs.shape)

	#net = self.apollo_net
	#images = "data_images" 
	#images_dropout = "data_images_dropout"
	#if images not in net.blobs:
	#	net.f(DummyData(images,image_data_rs.shape))
	#net.blobs[images].data[...] = images_data_rs 

	conv_proj_image = nn.Conv2d(in_channels=num_channels, out_channels= att_hidden, kernel_size=1)
	images = conv_proj_image(image_data_rs)
	print("Image after Convolution: ")
	print(images)

	#if dropout: 
	#	net.f(Dropout(images_dropout), 0.5, bottoms =[images]))
	# 	return images_dropout
	#else:
	#	return images

	if dropout:
		images=F.dropout(images, p=0.5, training = self.training)

	return images

#next step: forward lstm pt1: where all variables r def, pt 2: 


def main():
    file_path = 'data/images/COCO_val2014_000000000074.jpg'
    image_tensor = load_image_to_tensor(file_path)
    print(image_tensor.shape)
    #display_image_tensor(image_tensor)
    batch_image_tensor = image_tensor.unsqueeze(0)
    print(batch_image_tensor.shape)
    reshaped_image_tensor = forward_image_data(batch_image_tensor,False)
    print(reshaped_image_tensor.shape)
if __name__ == "__main__":
    main()






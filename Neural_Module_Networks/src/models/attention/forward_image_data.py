import torch
import numpy as np

batch_size = 1
num_channels = 3
image_height = 10
image_width = 10
image_data = torch.randn(batch_size,num_channels,image_width,image_height)
print(image_data)
print(image_data.shape)
image_data_rs = image_data.reshape((batch_size, num_channels, image_width, image_height,1))
print(image_data_rs)
print(image_data_rs.shape)

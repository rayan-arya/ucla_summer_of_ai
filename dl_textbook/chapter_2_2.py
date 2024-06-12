import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')


import pandas as pd

data = pd.read_csv(data_file)  #reading in the data file
print(data)

inputs, targets = data.iloc[:,0:2], data.iloc[:,2] #created input values and target values
inputs = pd.get_dummies(inputs, dummy_na =True) #for the input values - add column to indicate NaNs
print(inputs)

inputs = inputs.fillna(inputs.mean()) #replace the NaNs with the mean 
print(inputs)

import torch

tensorinput = torch.tensor(inputs.to_numpy(dtype = float))
tensoroutput = torch.tensor(targets.to_numpy(dtype=float))

print(tensorinput, tensoroutput)
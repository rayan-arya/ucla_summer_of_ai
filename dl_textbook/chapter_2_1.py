import torch

practice_tensor = torch.arange(12, dtype = torch.float32)

print(practice_tensor)
print(practice_tensor.numel())  #number of elements 
print(practice_tensor.size())  #size because the vector is one dimensional; otherwise would have been ".shape()"


reshaped_tensor = practice_tensor.reshape(3,4)  #reshape the vector
print(reshaped_tensor)

allzeros_tensor = torch.zeros((2,3,4))	#create two 3x4 tensors with all elements equal to 0
print(allzeros_tensor)

allones_tensor = torch.ones((3,4))	#create one 3x4 tensors with all elements equal to 1
print(allones_tensor)

randomtensor = torch.randn(3,4)
print(randomtensor)

preset_tensor = torch.tensor([[4,5,6], [5,7,2]])
print(preset_tensor)


##Indexing and Slicing

print(reshaped_tensor[-1], reshaped_tensor[1:3])
#the "-1" prints just the last row because using a negative takes from the end of a list
#the second tensor takes from row 1(not inclusive) to 3 (inclusive)

reshaped_tensor[1,2] = 12  #set index 1,2 to 12
print(reshaped_tensor)

reshaped_tensor[:3, :] = 12 #set multiple elements to the same thing
print(reshaped_tensor)


##Operations

operate_tensor = reshaped_tensor

exponentiated_tensor = torch.exp(operate_tensor) #torch.exp is e^x
print(exponentiated_tensor)

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2,2,2,2])

print(x+y, x-y, x*y, x/y, x**y) #** is exponentiation

new_tensor = torch.arange(12, dtype=torch.float32).reshape(3,4)
othernew_tensor = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
combined_tensor = torch.cat((new_tensor,othernew_tensor), dim=0), torch.cat((new_tensor,othernew_tensor),dim=1)
print(combined_tensor)
diffdimtensor = torch.cat((new_tensor,othernew_tensor), dim=1), torch.cat((new_tensor,othernew_tensor),dim=0)
print('diffdimtensor: ', diffdimtensor)
#combine two tensors - dim is the dimensions as an integer

bool_tensor = new_tensor == othernew_tensor #boolean
print(bool_tensor)

summed_tensor = new_tensor.sum()
print(summed_tensor)

## Broadcasting

first_reshape = torch.arange(3, dtype = torch.float32).reshape(3,1)  
second_reshape = torch.arange(2, dtype = torch.float32).reshape(1,2)

print(first_reshape, second_reshape) #expand arrays by copying elements along axes with length 1

print(first_reshape + second_reshape)

##Saving Memory 

memory_tensor = torch.tensor([[2.0,1,4,3],[1,2,3,4],[4,3,2,1]])
other_memory_tensor = new_tensor
before = id(memory_tensor)
memory_tensor = memory_tensor + other_memory_tensor
issame = id(memory_tensor) == before
print(issame)
#compare the ids of the two tensors

Z = torch.zeros_like(memory_tensor)  #zeros like - override values of tensor Z so that it is the same shape as Y
print('id(Z):',id(Z))
Z[:]= other_memory_tensor+memory_tensor
print('id(Z):', id(Z))


















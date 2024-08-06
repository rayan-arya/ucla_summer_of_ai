import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x+y, x*y, x/y, x**y) #basic scalar operations 

vectorx = torch.arange(3) #create vector

print(vectorx)

elementthree = vectorx[2] 
print(elementthree) #prints the third element of the vector, which is at index 2

vectorxlength = len(vectorx)
print(vectorxlength)

vectorxshape = vectorx.shape
print(vectorxshape)

matrixA = torch.arange(6).reshape(3,2) #create a matrix with values from 0 to 5 and then reshape the one dimensional vector into a 3x2 matrix
print(matrixA)

transposedmatrixA = matrixA.T
print(transposedmatrixA) #transposes the matrix(flipped axes)

symMatrix = torch.tensor([[1,2,3],[2,0,4],[3,4,5]])
print(symMatrix == symMatrix.T) #if a matrix is the same as its transposed matrix, it is symmetrical


tensorprac = torch.arange(24).reshape(2,3,4)
print(tensorprac) #tensor with values from 1-24 reshaped into 2 3x4 matricies

A = torch.arange(6, dtype=torch.float32).reshape(2,3)
B = A.clone() #creaitng a copy of A called B (new memory)
print(A, B, A+B, A*B) #when tensors with the same shape are multiplied, they produce a tensor with the same shape as the original tensor

reductiontensor = torch.arange(3, dtype = torch.float32)
print(reductiontensor, reductiontensor.sum()) #sum of all elements

print(A.shape, A.sum()) #sum function sums all values from all axes

print(A.shape, A.sum(axis=0).shape) #specifying axis = 0 will only sum that axis and will return a scalar

print(A.sum(axis=[0,1]) == A.sum()) #0-1 is range of rows, which encapsulates the entire matrix so it is the same as just taking the sum

meanoftensor = A.mean()

print(meanoftensor, A.sum()/A.numel()) #mean method is the same as sum/number of elements


sum_A = A.sum(axis = 1, keepdims=True) #keep axes unchanged at all times
print(sum_A, sum_A.shape)


cumulativeSum = A.cumsum(axis=0) #cumulative sum of an axis 
print(cumulativeSum)

dotproductout = torch.ones(3,dtype=torch.float32)
print(reductiontensor, dotproductout, torch.dot(reductiontensor,dotproductout)) #dotproduct

print(torch.sum(reductiontensor * dotproductout))
 
print(A.shape, reductiontensor.shape, torch.mv(A, reductiontensor), A@reductiontensor) #prints the matrix-vector product (both .mv and @ work)

B = torch.ones(3,4)
print(torch.mm(A,B), A@B) #matrix multiplication

u = torch.tensor([3.0, -4.0])
print(torch.norm(u)) #L2  norm is the euclidian length of a vector
print(torch.abs(u).sum()) #L1 norm is the manhattan distance
print(torch.norm(torch.ones(4,9))) #Lp norm is the forbenius norm

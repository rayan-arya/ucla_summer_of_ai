import torch

practiceTensor = torch.arange(4.0) #create vector
print(practiceTensor)

practiceTensor.requires_grad_(True) #if gradients are going to be computed - avoid allocating new memory for each Dv
print(practiceTensor.grad)

outputTensor = 2*torch.dot(practiceTensor,practiceTensor)
print(outputTensor) #calculate function and assign it to output tensor 

outputTensor.backward()#take the gradient of outputTensor in respect to practiceTensor
print(practiceTensor.grad)

print(practiceTensor.grad == 4*practiceTensor) #check if the function is correct because we know that the gradient of y=2x . x with respect to x is 4x

practiceTensor.grad.zero_() #reset gradient before you change functions
outputTensor = practiceTensor.sum()
outputTensor.backward()
print(practiceTensor.grad) 

practiceTensor.grad.zero_()
outputTensor=practiceTensor*practiceTensor
outputTensor.backward(gradient=torch.ones(len(outputTensor))) #backward on non scalar gives an error, so use this method instead
print(practiceTensor.grad)

practiceTensor.grad.zero_()
outputTensor = practiceTensor*practiceTensor
u = outputTensor.detach() #use when you want to do calculations that are not to be included while computing a gradient
z = u * practiceTensor

z.sum().backward()
print(practiceTensor.grad == u)

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()

print(a.grad == d/a) #dynamic flow - not just fixed equations like the previous examples. 

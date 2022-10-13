###Episode 1

import numpy as np
import torch

#Training data

# Input (temp, rainfall, humidity)
inputs = np.array([[73,67,43], [91,88,64], [87, 134, 58], [108, 43, 37], [69, 96,70]], dtype='float32')


# targets (apples, oranges)
targets = np.array([[56, 70], [81,101], [119, 133], [22,37], [103, 119]], dtype = 'float32')

#Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print (inputs)
print(targets)

###Linear Regression model

#Weights and Biases
w = torch.randn(2, 3, requires_grad = True)
b = torch.randn(2, requires_grad = True)
print(w)
print(b)

###Model
def model(x):
    return x @w.t() + b
    
# Generate predictions
preds = model(inputs)
print(preds)

#Compare with actual targets
print(targets)


### Loss Function : to check how well our model is performing
    # Calculate the difference betweeen the two matrices (preds and targets).
    # Square all the elements of the difference matrix to remove negative values.
    # Calaculate the average of the elements in the resulting matrix.

#diff = preds - targets
#print(diff)
#diff_sqr = diff * diff
#torch.sum(diff__sqr)/diff.numel()

#Mean Squared Error (MSE) loss
def mse(t1, t2):
   diff = t1 - t2 
   return torch.sum(diff * diff) / diff.numel()

# Compute Loss
loss = mse(preds, targets)
print(loss)


# Compute Gradients
loss.backward()

# Gradients for weights
print(w)
print(w.grad)

# Gradients for biases
print(b)
print(b.grad)


# Reset the values of Grad values NOT the actual values
w.grad.zero_()
b.grad.zero_()
print(w)
print(b)

### Adjust weights and biases using gradient descent
    # Generate predictions    
    # Calculate the loss
    # Compute gradients w.r.t weights and biases
    # Adjust the weights by subtracting a small quantity proportional to the gradient
    # Reset gradients to zero

# Generate predictions
preds = model(inputs)
print(preds)

# Calculate the loss
loss = mse (preds, targets)
print(loss)

# Compute Gradients
loss.backward()
print(w.grad)
print(b.grad)

# Adjust the weights and rest gradients
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()

print(w)
print(b)

#Lets check if the above slight change in the weights have improved the loss value (i.e decrease) or not.

#Calculate loss again
preds = model(inputs)
loss = mse (preds, targets)
print(loss)


# Train for multiple epochs (let say for 500 epochs)

for i in range (500):
    preds =model(inputs)
    loss = mse(preds, targets)
    loss.backward()  # To calculate the gradients 
    with torch.no_grad():
        w -= w.grad * 1e-5   # 1e-5 is Learning rate which is a hyper-parameter in machine learning
        b -= b.grad * 1e-5  
        w.grad.zero_()       # Reset the gradients to zero
        b.grad.zero_()
        
### Lets, calculate the loss again
# Calculate loss
preds = model(inputs)
loss = mse (preds, targets)
print(loss)

### Lets compare predictions and targets(actual values)
#Predictions            
print(preds)

#Targets (Actual values)
print(targets)

##### Working with Jovian 
##Intall
#pip install jovian --upgrade -q
#import jovian
#jovian.commit()

 
################# Linear regression using PyTorch built-insert without making manual functions
import numpy as np
import torch
import torch.nn as nn

# Input (temp, rainfall, humidity)
inputs = np.array([[73,67,43], [91, 88, 64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91,88,64], [87, 134, 58], [102, 43, 37], [69, 96, 70], [73, 67, 43], [91, 88, 64],[87, 134, 58], [102, 43, 37], [69, 96, 70]], dtype = 'float32')

#targets (apples, oranges)
targets =np.array([[56, 70], [81, 101], [119, 133], [22, 37], [103, 119], [56, 70], [81, 101], [119, 133], [22, 37], [103, 119], [56, 70], [81, 101], [119, 133], [22, 37], [103, 119]], dtype = 'float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)

###Dataset and dataloader
# We are not going to use complete dataset but in batches to deal with memory issues and less complex computations

from torch.utils.data import TensorDataset

# Define Dataset using TensorDataset
train_ds = TensorDataset(inputs, targets)
train_ds[0:3]
# You can also pick specific rows of data)
#train_ds[[1, 3, 5, 7]]

from torch.utils.data import DataLoader

#Define DataLoader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

for xb, yb in train_dl:
    #print('batch:')  # prints all the batches
    print(xb)
    print(yb)
    break           # use break if you want to work only one batch and remove the line 179 that prints all batches

# Define model using nn.Linear
model = nn.Linear (3, 2)
print(model.weight)
print(model.bias)

# Parameters
list(model.parameters())

#Generate predictions
preds = model(inputs)
print(preds)

###Loss Functions
# Import nn.functional
import torch.nn.functional as F

# Define Loss functional
loss_fn = F.mse_loss
loss = loss_fn(model(inputs),targets)
print(loss)

# Note: to read help on Linear model of pytorch use following line
# ?nn.Linear
# ?F.mse_loss

###Optimizer
# Define optimizer (Stochastic Gradient Descent)
opt = torch.optim.SGD(model.parameters(), lr=1e-5)

### Train the model
    # Generate predictions    
    # Calculate the loss
    # Compute gradients w.r.t weights and biases
    # Adjust the weights by subtracting a small quantity proportional to the gradient
    # Reset gradients to zero
    
#Utility fiunction to train the model
def fit(num_epochs, model, loss_fn, opt):

    #Repeat for given number of epochs
    for epoch in range(num_epochs):
    
        #Train with batches of data
        for xb, yb in train_dl:
        
            #1. Generate predictions
            pred = model(xb)
            
            #2. Calculate loss
            loss = loss_fn(pred, yb)
            
            #3. Compute gradients
            loss.backward()
            
            #4. Update parameters using gradients
            opt.step()
            
            #5. reset the gradients to zero
            opt.zero_grad()
            
        #Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item())) 
            # loss.item gives the actual value of th eloss in that batch (here after every 10th epoch)
            
# Train the model for 100 epochs
fit(100, model, loss_fn, opt)            

# Generate predictions
preds = model(inputs)
print(preds)

# Compare the weights
print(targets)

### Commit and update the notebook
import jovian
jovian.commit()
            


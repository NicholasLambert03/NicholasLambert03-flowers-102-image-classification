#imports
import torch
import torchvision
#following import name conventions
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

#processor choice
#Use GPU if possible
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
# From https://pytorch.org/tutorials/beginner/nn_tutorial.html
class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield (self.func(*b))

def preprocess(x, y):
    return x.to(device), y.to(device)  

#Loading and normalising data
#Transform to be ran on data
train_transforms = transforms.Compose([
    
    transforms.RandomResizedCrop((256,256)),
    transforms.RandomPerspective(),
    transforms.RandomHorizontalFlip(), #reduce bias with image flip
    transforms.RandomRotation(30), #reduce bias with random angles
    transforms.ColorJitter(brightness=0.5,contrast=0.5,hue=0.2),

    
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #mean and std of dataset
])

test_transforms = transforms.Compose([
    transforms.Resize((256,256)),

    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) #mean and std of dataset
])

#Hyperparameters
batch_size = 32
learning_rate = 1e-2
epoch_count = 1000
val_batch_size = 32
weight_decay = 3e-4
breakout_percentage = 2e-1
dropout = 0.5

train_data = torchvision.datasets.Flowers102(
    root="./data",
    download=True,
    split= "train",
    transform=train_transforms
)
validation_data = torchvision.datasets.Flowers102(
    root="./data",
    download=True,
    split= "val",
    transform=test_transforms
)
test_data = torchvision.datasets.Flowers102(
    root="./data",
    download=True,
    split= "test",
    transform=test_transforms
)

train_loader = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,batch_size=batch_size)
validation_loader = torch.utils.data.DataLoader(validation_data,val_batch_size,drop_last=True,)


train_loader = WrappedDataLoader(train_loader, preprocess)
test_loader = WrappedDataLoader(test_loader, preprocess)
validation_loader = WrappedDataLoader(validation_loader, preprocess)

#CNN

class cnn(nn.Module):
    def __init__(self):
        #Layers
        
        super().__init__()
        self.conv1 = nn.Conv2d(3,16,3)# params are input channels, output channels, filter(kernel) size
        self.batchnorm1 = nn.BatchNorm2d(16) 
        self.pool1 = nn.MaxPool2d(2,2) # params are kernal size, stride

        self.conv2 = nn.Conv2d(16,32, 3)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2,2)

       
        self.conv3 = nn.Conv2d(32,64, 3)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2,2)


        self.lin1 = nn.Linear(57600,1000)
        self.lin2 = nn.Linear(1000,102) #output features must be equal to num of categories
        self.dropout = nn.Dropout(dropout) #Randomly sets input features to zero to counter overfitting
    
    def forward(self,x):
        """Called to pass input data through layers, uses activation function"""
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x= F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x= F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batchnorm3(x)
        x= F.relu(x)
        x= self.pool3(x)

        x = torch.flatten(x, 1)

        x = self.lin1(x)
        x= F.relu(x)
        x = self.dropout(x)


        x = self.lin2(x)
    
        return x

model = cnn().to(device)

#loss function and optimizer

loss_fn = nn.CrossEntropyLoss() #  probibalistic good for multiclass uses softmax
optimizer = optim.SGD(model.parameters(),learning_rate,weight_decay=weight_decay) #Stochastic gradient descent

#training loop
escape_loop = False
lowest_val_loss = 10.0
val_previous_loss=10.0
for epoch in range(epoch_count): #loop multiple times
    model.train()
    compare_loss = 0.0
    running_loss = 0.0
    val_running_loss = 0.0
    val_final_loss = 0.0
    breakout_loss = 0.0
    for batch, (images,labels) in enumerate(train_loader): # pulls out inputs and labels from training data)
        optimizer.zero_grad() #zero parameter gradients
        
        #forward cnn, backpropagation (backward and optimize)
        outputs = model(images)
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimizer.step()
        
        #print statistics
        running_loss += loss.item()
        if((batch+1)%8 == 0):
            print(f"[Epoch: {epoch + 1}, Batch progress: {batch + 1:5d}] loss: {running_loss/8:.3f}") 
            compare_loss += running_loss/8
            running_loss = 0.0

    #Each epoch calculate validation loss as to know if to break out due to overfitting
    model.eval()
    with torch.no_grad(): # in validation, don't need to calculate gradients for ouputs
        correct = 0
        total = 0
        val_acc = 0
        for i,(val_images,val_labels) in enumerate(validation_loader):
            val_outputs = model(val_images)
            _, predicted = torch.max(val_outputs.data,1) # returns predicted class labels in one dimension
            total += val_labels.size(0) #adds number of samples in batch to total sample count
            correct += (predicted == val_labels).sum().item()
            val_loss = loss_fn(val_outputs,val_labels)
            val_running_loss += val_loss.item()
            val_acc = 100 * correct / total
        val_final_loss = val_running_loss/(i+1)
        print(f"Validation loss: {val_final_loss:.3f} Accuracy: {val_acc:.5f}%")
    
    #Breakout clause
    #if validation loss is > loss + breakout% exit
    compare_loss /=4
    breakout_loss = (compare_loss + (compare_loss *breakout_percentage))
    print(f"Breakout if loss > {breakout_loss:.3f} and increased from {val_previous_loss:.3f}")
    if((val_final_loss > breakout_loss)and(val_final_loss > val_previous_loss)):
        print("Broken out")
        break
    if val_final_loss < lowest_val_loss: # Saves best fit model rather than model where begins to get worse
       best_model = copy.deepcopy(model.state_dict())
       lowest_val_loss = val_final_loss
       print("NEW BEST")
    val_previous_loss = val_final_loss
    
print("Training complete")

#test model




correct = 0
total = 0
accuracy = 0
model.load_state_dict(best_model)
#CODE TO LOAD SAVED MODEL UNCOMMENT FOR USE
#model.load_state_dict(torch.load(models\model_accuracy_46.90194.pt)) 
model.eval()
with torch.no_grad(): # in testing, don't need to calculate gradients for ouputs
    for (images,labels) in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data,1) # returns predicted class labels in one dimension
        total += labels.size(0) #adds number of samples in batch to total sample count
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
print(f"Accuracy of the network on the test images: {test_accuracy:.5f} %")

#save model
torch.save(best_model, f"models/model_accuracy_{test_accuracy:.5f}.pt")
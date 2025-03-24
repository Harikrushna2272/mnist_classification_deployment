import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.nn as nn


if torch.cuda.is_available():
    device=torch.device(type="cuda",index=0)
else:
    device=torch.device(type="cpu",index=0)


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3881,))])
train_dataset=datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform,
)
test_dataset=datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform,
)



batch_size=64

train_dl=DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_dl=DataLoader(
    dataset=test_dataset,
    batch_size=batch_size, 
    shuffle=True
)

class MNISTNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.inh1=nn.Linear(in_features=784, out_features=512)
        self.relu=nn.ReLU()
        self.bn1=nn.BatchNorm1d(num_features=512)
        self.h2=nn.Linear(in_features=512, out_features=256)
        self.bn2=nn.BatchNorm1d(num_features=256)
        self.h3=nn.Linear(in_features=256, out_features=128)
        self.bn3=nn.BatchNorm1d(num_features=128)
        self.h4=nn.Linear(in_features=128, out_features=64)
        self.bn4=nn.BatchNorm1d(num_features=64)
        self.h5=nn.Linear(in_features=64, out_features=32)
        self.bn5=nn.BatchNorm1d(num_features=32)
        self.output=nn.Linear(in_features=32, out_features=10)
        self.bn6=nn.BatchNorm1d(num_features=10)
        
    def forward(self,x):
        x=self.inh1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.h2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.h3(x)
        x=self.bn3(x)
        x=self.relu(x)
        x=self.h4(x)
        x=self.bn4(x)
        x=self.relu(x)
        x=self.h5(x)
        x=self.bn5(x)
        x=self.relu(x)
        x=self.output(x)
        output=self.bn6(x)
        return output
    
def train_one_epoch(dataloader, model,loss_fn, optimizer):
    model.train()
    track_loss=0
    num_correct=0
    for i, (imgs, labels) in enumerate(dataloader):
        imgs=torch.reshape(imgs,shape=[-1,784]).to(device)
        labels=labels.to(device)
        pred=model(imgs)
                    
        loss=loss_fn(pred,labels)
        track_loss+=loss.item()
        num_correct+=(torch.argmax(pred,dim=1)==labels).type(torch.float).sum().item()
        
        running_loss=round(track_loss/(i+(imgs.shape[0]/batch_size)),2)
        running_acc=round((num_correct/((i*batch_size+imgs.shape[0])))*100,2)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i%100==0:
            print("Batch:", i+1, "/",len(dataloader), "Running Loss:",running_loss, "Running Accuracy:",running_acc)
            
    epoch_loss=running_loss
    epoch_acc=running_acc
    return epoch_loss, epoch_acc

def eval_one_epoch(dataloader, model,loss_fn):
    model.eval()
    track_loss=0
    num_correct=0
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(dataloader):
            imgs=torch.reshape(imgs,shape=[-1,784]).to(device)
            labels=labels.to(device)
            pred=model(imgs)
            loss=loss_fn(pred,labels)
            track_loss+=loss.item()
            num_correct+=(torch.argmax(pred,dim=1)==labels).type(torch.float).sum().item()
            running_loss=round(track_loss/(i+(imgs.shape[0]/batch_size)),2)
            running_acc=round((num_correct/((i*batch_size+imgs.shape[0])))*100,2)
            
            if i%100==0:
                print("Batch:", i+1, "/",len(dataloader), "Running Loss:",running_loss, "Running Accuracy:",running_acc)

    epoch_loss=running_loss 
    epoch_acc=running_acc
    return epoch_loss, epoch_acc

model=MNISTNN()
model=model.to(device)
loss_fn=nn.CrossEntropyLoss()
lr=0.001
#optimizer=torch.optim.SGD(params=model.parameters(), lr=lr)
optimizer=torch.optim.Adam(params=model.parameters(), lr=lr)
n_epochs=3

for i in range(n_epochs):
    print("Epoch No:",i+1)
    train_epoch_loss, train_epoch_acc=train_one_epoch(train_dl,model,loss_fn,optimizer)
    val_epoch_loss, val_epoch_acc=eval_one_epoch(test_dl,model,loss_fn)
    print("Training:", "Epoch Loss:", train_epoch_loss, "Epoch Accuracy:", train_epoch_acc)
    print("Inference:", "Epoch Loss:", val_epoch_loss, "Epoch Accuracy:", val_epoch_acc)
    print("--------------------------------------------------")

torch.save(model.state_dict(), "model_mnist.pth")
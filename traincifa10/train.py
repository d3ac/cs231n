import torch
from torch import nn
from data_processing import get_dataiter, get_class_names, get_data
from torch.nn import functional as F
import d2l.torch as d2l
from model.ResNet18 import ResNet18, ResBlock
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def get_correct_n(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().sum().item()

def evaluate_accuracy(model, data_iter):
    model.eval()
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        for X, y in data_iter:
            X, y = X.to(torch.device('cuda')), y.to(torch.device('cuda'))
            acc_sum += get_correct_n(model(X), y)
            n += y.shape[0]
        return acc_sum / n

def train(model, train_iter, test_iter, loss, optimizer, num_epochs, batch_size):
    cudnn.benchmark = True
    Trange = tqdm.tqdm(range(num_epochs))
    train_acc, test_acc = [], []
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150, 175], gamma=0.5)
    best_acc = 0.91
    for epoch in Trange:
        model.train()
        acc_n = 0
        n = 0
        for X, y in train_iter:
            X, y = X.to(torch.device('cuda')), y.to(torch.device('cuda'))
            y_hat = model(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            acc_n += get_correct_n(y_hat, y)
            n += y.shape[0]
        scheduler.step()
        train_acc.append(acc_n / n)
        if epoch % 5 == 0:
            test_acc.append(evaluate_accuracy(model, test_iter))
        else:
            test_acc.append(test_acc[-1])
        if test_acc[-1] > best_acc:
            best_acc = test_acc[-1]
            torch.save(model.state_dict(), 'model.pth')
        Trange.set_description(f'epoch {epoch+1}, train acc {train_acc[-1]:.4f}, test acc {test_acc[-1]:.4f}')
    return train_acc, test_acc

if __name__ == '__main__':
    # Set the hyperparameters
    batch_size = 128
    momentum = 0.9
    lr = 0.1
    weight_decay = 1e-4 
    epochs = 200
    resize = 64

    # Load the data
    train_iter = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.RandomCrop(np.random.randint(resize*0.8, resize)),
            transforms.Resize((resize, resize)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.03), ratio=(0.3, 1), value=0, inplace=False),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.03), ratio=(0.3, 1), value=0, inplace=False),
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=7, pin_memory=True)
    
    test_iter = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=7, pin_memory=True)

    # get the model
    model = ResNet18(ResBlock).to(torch.device('cuda'))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()
    train_acc, test_acc = train(model, train_iter, test_iter, loss, optimizer, epochs, batch_size)
    
    # Plot the results
    plt.plot(np.arange(1, epochs+1), train_acc, label='train acc', color='red')
    plt.plot(np.arange(1, epochs+1), test_acc, label='test acc', color='blue')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
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

def test(model, test_iter, batch_size):
    cudnn.benchmark = True
    test_acc = evaluate_accuracy(model, test_iter)
    return test_acc

if __name__ == '__main__':
    # Set the hyperparameters
    batch_size = 128
    momentum = 0.9
    lr = 0.1
    weight_decay = 1e-4 
    epochs = 200
    resize = 64

    # Load the data    
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
    model.load_state_dict(torch.load('model.pth', weights_only=True))
    test_acc = test(model, test_iter, batch_size)
    print(f'test acc {test_acc}')
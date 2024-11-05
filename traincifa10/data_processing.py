import numpy as np
import random
from torchvision.transforms import transforms
import imgaug
import torch
import cv2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_dataset(filename):
    labels = []
    features = None
    for file in filename:
        data = unpickle(file)
        labels_temp = data[b'labels']
        features_temp = data[b'data']
        labels.extend(labels_temp)
        if features is None:
            features = features_temp
        else:
            features = np.vstack((features, features_temp))
    return features, np.array(labels)

def data_augmentation(features, resize, train):
    # random crop
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(np.random.randint(resize*0.8, resize)),
        transforms.Resize((resize, resize)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.03), ratio=(0.3, 1), value=0, inplace=False),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.03), ratio=(0.3, 1), value=0, inplace=False),
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    if train:
        features = torch.stack([transform_train(img.transpose(1, 2, 0)) for img in features])
    else:
        features = torch.stack([transform_test(img.transpose(1, 2, 0)) for img in features])
    return features


def get_data(batch_size, resize=None):
    # train
    train_features, train_labels = read_dataset(['data/data_batch_1', 'data/data_batch_2', 'data/data_batch_3', 'data/data_batch_4', 'data/data_batch_5'])
    if resize is not None:
        train_features = np.array([cv2.resize(img.reshape(3, 32, 32).transpose(1, 2, 0), (resize, resize)).transpose(2, 0, 1) for img in train_features])
    mean = np.mean(train_features, axis=(0, 2, 3))
    std = np.std(train_features, axis=(0, 2, 3))
    train_features = (train_features - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)
    # test
    test_features, test_labels = read_dataset(['data/test_batch'])
    if resize is not None:
        test_features = np.array([cv2.resize(img.reshape(3, 32, 32).transpose(1, 2, 0), (resize, resize)).transpose(2, 0, 1) for img in test_features])
    test_features = (test_features - mean.reshape(1, 3, 1, 1)) / std.reshape(1, 3, 1, 1)

    return train_features, train_labels, test_features, test_labels

def get_dataiter(features, labels, batch_size, resize=None, train=True):
    features = data_augmentation(features, resize, train)
    dataset = torch.utils.data.TensorDataset(features, torch.tensor(labels, dtype=torch.long))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train)
    return data_loader

def get_class_names():
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return labels

def sample_one_img(features, labels):
    index = random.randint(0, features.shape[0])
    return features[index].reshape(3, 32, 32).transpose(1, 2, 0), labels[index]


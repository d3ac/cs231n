import numpy as np
import random
import imgaug
import torch

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_train_dataset():
    filename = ['data/data_batch_1', 'data/data_batch_2', 'data/data_batch_3', 'data/data_batch_4', 'data/data_batch_5']
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

def get_train_dataiter(batch_size, resize=None):
    features, labels = read_train_dataset()
    if resize is not None:
        features = np.array([imgaug.augmenters.Resize(resize).augment_image(img.reshape(3, 32, 32).transpose(1, 2, 0)).transpose(2, 0, 1).flatten() for img in features])
    dataset = torch.utils.data.TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader

def read_test_dataset():
    data = unpickle('data/test_batch')
    return data[b'data'], np.array(data[b'labels'])

def get_class_names():
    labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    return labels

def sample_one_img(features, labels):
    index = random.randint(0, features.shape[0])
    return features[index].reshape(3, 32, 32).transpose(1, 2, 0), labels[index]


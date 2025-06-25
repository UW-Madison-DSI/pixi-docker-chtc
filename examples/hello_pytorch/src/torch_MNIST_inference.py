# c.f. https://www.kaggle.com/code/geekysaint/solving-mnist-using-pytorch#Training-the-Model
# c.f. https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from pathlib import Path
import matplotlib.pyplot as plt


# Use the CNN architecture defined in torch_MNIST.py
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


data_dir = Path(Path(__file__).parents[1] / "data").resolve()

# test_dataset = MNIST(root = 'data/', train = False, transform = transforms.ToTensor())

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

# Model class must be defined somewhere
model_path = Path(Path(__file__).parents[1] / "mnist_cnn.pt").resolve()
state_dict = torch.load(model_path, weights_only=True)

# Remember that you must call model.eval() to set dropout and batch normalization
# layers to evaluation mode before running inference.
# Failing to do this will yield inconsistent inference results.
model = Net()
model.load_state_dict(state_dict)
model.eval()

img, label = test_dataset[0]
plt.imshow(img[0], cmap="gray")
# plt.show()
print("shape: ", img.shape)
print("Label: ", label)

print(img.unsqueeze(0).shape)
print(img.shape)


def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


img, label = test_dataset[250]
plt.imshow(img[0], cmap="gray")
plt.show()
print("Label:", label, ",Predicted:", predict_image(img, model))

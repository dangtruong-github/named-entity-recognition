import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleCNN(nn.Module):
    def __init__(self, in_channels = 32, num_classes = 18):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1) 
        # convolution with kernel 3x3, stride 1x1, and padding 1x1
        
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # max pooling of kernel 2x2, stride 2x2
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) 
        
        self.fc1 = nn.Linear(128*17, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        
        return x
    
if __name__ == "__main__":
    cnn = SimpleCNN()
    data = torch.rand((32, 32, 70))

    out = cnn(data)
    print(out.shape)
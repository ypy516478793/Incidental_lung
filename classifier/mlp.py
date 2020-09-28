import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, input_dim, output_dim=1, num_features=256):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2*num_features)
        self.fc2 = nn.Linear(2*num_features, num_features)
        self.fc3 = nn.Linear(num_features, output_dim)
        # self.drop1 = nn.Dropout(p=0.01)
        # self.drop2 = nn.Dropout(p=0.01)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = self.drop1(x)
        x = F.relu(self.fc2(x))
        # x = self.drop2(x)
        x = self.fc3(x)
        return x

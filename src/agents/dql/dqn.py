import torch.nn as nn


class DQN(nn.Module):

    def __init__(self, state_length, action_length):
        super(DQN, self).__init__()
        self.policy_network = nn.Sequential(
            nn.Linear(state_length, 180),
            nn.BatchNorm1d(180),
            nn.Linear(180, 360),
            nn.BatchNorm1d(360),
            nn.Linear(360, action_length))

        # self.layer1 = nn.Linear(state_length, 128)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.layer2 = nn.Linear(128, 256)
        # self.bn2 = nn.BatchNorm1d(256)
        # self.out = nn.Linear(256, action_length)

    def forward(self, x):
        # if x.shape[0] > 1:
        #     x = F.relu(self.bn1(self.layer1(x)))
        #     x = F.relu(self.bn2(self.layer2(x)))
        # else:
        #     x = F.relu(self.layer1(x))
        #     x = F.relu(self.layer2(x))
        # return self.out(x)
        return self.policy_network(x)

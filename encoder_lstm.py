import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (h_n, c_n) = self.lstm(x)
        return h_n, c_n
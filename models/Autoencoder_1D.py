import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, dropout):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=12, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout(p=dropout)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout(p=dropout)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.batch_norm4 = nn.BatchNorm1d(12)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout4 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout1(self.pool1(self.relu1(self.batch_norm1(self.conv1(x)))))
        x = self.dropout2(self.pool2(self.relu2(self.batch_norm2(self.conv2(x)))))
        x = self.dropout3(self.pool3(self.relu3(self.batch_norm3(self.conv3(x)))))
        x = self.dropout4(self.pool4(self.relu4(self.batch_norm4(self.conv4(x)))))
        return x


class Decoder(nn.Module):
    def __init__(self, dropout):
        super(Decoder, self).__init__()
        self.transpose_conv1 = nn.ConvTranspose1d(12, 64, kernel_size=2, stride=2, padding=0)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        self.transpose_conv2 = nn.ConvTranspose1d(64, 128, kernel_size=2, stride=2, padding=0)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout)

        self.transpose_conv3 = nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2, padding=0)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout)

        self.transpose_conv4 = nn.ConvTranspose1d(64, 12, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.dropout1(self.relu1(self.batch_norm1(self.transpose_conv1(x))))
        x = self.dropout2(self.relu2(self.batch_norm2(self.transpose_conv2(x))))
        x = self.dropout3(self.relu3(self.batch_norm3(self.transpose_conv3(x))))
        x = self.transpose_conv4(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, dropout=0.2):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(dropout)
        self.decoder = Decoder(dropout)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == '__main__':
    autoencoder = Autoencoder(dropout=0.2)
    x = torch.randn((16, 12, 512))
    decoded = autoencoder(x)
    print(decoded.shape)

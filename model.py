import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, inpput_dims, encoding_dims):
        super(Autoencoder, self).__init__()
        
        # TODO: This is fairly basic: can we do better?
        self.encoder = nn.Sequential(
            nn.Linear(inpput_dims, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dims),
        )

        self.decoder = nn.Sequential(
            nn.Linear(encoding_dims, 128),
            nn.ReLU(),
            nn.Linear(128, inpput_dims)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
import torch

# TODO: Right now, hyperparams are fixed.
# There's room to add a wrapper which tunes the hyperparams
# according to performance.
def train_autoencoder(model, data, epochs = 100,
                      batch_size = 32, learn_rate = 0.001):
    # Optimization which attempts to use the GPU for calculations.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = torch.tensor(data, dtype = torch.float32).to(device)

    data = torch.tensor(
        data,
        dtype = torch.float32
    ).to(device)

    dataloader = torch.utils.data.DataLoader(
        data,
        batch_size = batch_size,
        shuffle = True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr = learn_rate)
    criterion = torch.nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            _, decoded = model(batch)
            loss = criterion(decoded, batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print("Epoch:", epoch + 1, "Loss:", loss.item())

    return model

def get_embeddings(model, data):
  model.eval()
  with torch.no_grad():
      data = torch.tensor(data, dtype = torch.float32).to(model.encoder[0].weight.device)
      embeddings, _ = model(data)

  return embeddings.cpu().numpy()
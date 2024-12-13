import torch
from sklearn.cluster import KMeans
import torch.nn.functional as F
from sklearn.cluster import KMeans

def get_embeddings(model, data):
    model.eval()  #set the model to evaluation mode
    with torch.no_grad():  #not computing gradients
        data = torch.tensor(data, dtype=torch.float32).to(model.encoder[0].weight.device) 
        embeddings, _ = model(data)  #get embeddings from the encoder
    return embeddings.cpu().numpy()

def train(model, clustering_layer, data, epochs, batch_size, learn_rate, n_clusters, device):
    optimizer = torch.optim.Adam(list(model.parameters()) + list(clustering_layer.parameters()), lr=learn_rate)
    criterion = torch.nn.MSELoss()
    data_tensor = torch.tensor(data, dtype=torch.float32).to(device) #moved to device, just once

    # Initialize cluster centers
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto') #n_init will prevent warning message for certain datasets
    kmeans.fit(get_embeddings(model, data))  #fit k-means on the initial embeddings
    clustering_layer.clusters.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

    model.train()
    clustering_layer.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in torch.utils.data.DataLoader(data_tensor, batch_size=batch_size, shuffle=True):
            optimizer.zero_grad()
            encoded, decoded = model(batch)
            q = clustering_layer(encoded)
            p = target_distribution(q).detach()
            kl_loss = F.kl_div(q.log(), p, reduction='batchmean')  #corrected KL divergence calculation
            reconstruction_loss = criterion(decoded, batch)
            loss = kl_loss + reconstruction_loss #added 
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        average_loss = total_loss / len(data_tensor)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {average_loss:.4f}")

    return model, clustering_layer #trained model is retuenred

def target_distribution(q):
    weight = q**2 / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()
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

def train(model, clustering_layer, data, epochs, batch_size, learn_rate, n_clusters):
    optimizer = torch.optim.Adam(list(model.parameters()) + list(clustering_layer.parameters()), lr=learn_rate)
    criterion = torch.nn.MSELoss()  #for reconstruction loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    clustering_layer.to(device)
    data = torch.tensor(data, dtype=torch.float32).to(device)

    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    y_pred = kmeans.fit_predict(get_embeddings(model, data.cpu().numpy()))  #init cluster centers
    clustering_layer.clusters.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

    model.train()
    clustering_layer.train()
    for epoch in range(epochs):
        for batch in torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True):

            optimizer.zero_grad()
            encoded, decoded = model(batch)
            q = clustering_layer(encoded)
            p = target_distribution(q).detach()  #target distribution
            kl_loss = F.kl_div(p.log(), q, reduction='batchmean')
            reconstruction_loss = criterion(decoded, batch)
            loss = kl_loss + reconstruction_loss
            loss.backward()
            optimizer.step()
        

def target_distribution(q):
    weight = q**2 / torch.sum(q, dim=0)
    return (weight.t() / torch.sum(weight, dim=1)).t()
import torch
import torch.nn as nn
import numpy as np
from utils.utils import kl_recon_loss, flatten, ROC
import copy 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

default_dim = {
'swat': 128,
'wadi': 128, 
'smap': 64,
'msl': 64,
'smd': 64,
'ucr': 64,
}

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )        

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

        self.meanfc = nn.Linear(hidden_dim, latent_dim)
        self.varfc = nn.Linear(hidden_dim, latent_dim)

    def reparametize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        noise = torch.randn_like(std).to(device)

        z = mu + noise * std
        return z

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape

        encoded = self.encoder(x)

        # extract latent variable z(hidden space to latent space)
        mean = self.meanfc(encoded)
        logvar = self.varfc(encoded)
        z = self.reparametize(mean, logvar)  # batch_size x latent_size        

        decoded = self.decoder(z)
        return decoded, mean, logvar  
    
    
def train(model, optimizer, train_loader, loss_fn):
    model.train()
    losses = [] 

    for batch in train_loader:
        x = batch.float().to(device)
        
        y_hat, m, v = model(x) 
        loss = loss_fn(x, y_hat, m, v)        

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses += [loss.item()]
    
    return losses 

def val(model, val_loader, loss_fn):
    model.eval()
    losses = []    

    for batch in val_loader:
        x = batch.float().to(device)

        y_hat, m, v = model(x) 
        loss = loss_fn(x, y_hat, m, v)
        
        losses += [loss.item()]
        
    return np.mean(losses)


def test(model, test_loader, loss_fn):       
    with torch.no_grad():
        labels = []
        scores = []
        
        model.eval()        
        for batch in test_loader:
            x, y = batch[0].float().to(device), batch[1].float().to(device)

            y_hat, m, v = model(x)       
            loss = loss_fn(x, y_hat, m, v)
  
            mse = nn.functional.mse_loss(x, y_hat, reduction='none').cpu()
            score = torch.mean(mse, 2).tolist()
            
            scores += score                
            labels += y.cpu().tolist()  

    return flatten(scores), flatten(labels)


def vae_experiment(train_loader, val_loader, test_loader, args):
    model_name = "vae" 
    print("Evaluating " + model_name)
    
    hidden_dim = args['hidden_dim']
    if  hidden_dim == 'default':
        hidden_dim = default_dim[args['dataset']]
    
    model = VAE(args['input_dim'],hidden_dim,12).to(device)
    loss_fn = kl_recon_loss
    optimizer = torch.optim.Adam(model.parameters())

    best_val = 10000

    for i in range(args['num_epochs']):
        train(model, optimizer, train_loader, loss_fn)
        val_losses = val(model, val_loader, loss_fn)
        if args['verbose']:
            print('Epoch {:d} Val Loss: {:f}'.format(i,val_losses))
        if val_losses < best_val:
            best_val = val_losses 
            best_model_state_dict = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), args['experiment_dir'] + "/" + model_name + ".pth")


    model.load_state_dict(torch.load(args['experiment_dir'] + "/" + model_name + ".pth"))
    scores, labels = test(model, test_loader,loss_fn)
    
    thresh, auc = ROC(labels, scores)
    metrics_dict = {
    "model" : [model_name], 
    "metric" : ["auc-roc"],
    "score" : [auc]}
    
    return metrics_dict

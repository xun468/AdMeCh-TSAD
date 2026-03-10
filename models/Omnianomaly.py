import torch
import torch.nn as nn
import copy
import numpy as np
from utils.utils import kl_recon_loss, flatten, ROC
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OmniAnomaly(nn.Module):
    def __init__(self, feats, hidden, latent):
        super(OmniAnomaly, self).__init__()
        self.name = 'OmniAnomaly'
        self.lr = 0.002
        self.beta = 0.01
        self.n_feats = feats
        self.n_hidden = hidden
        self.n_latent = latent
        self.lstm = nn.GRU(feats, self.n_hidden, 2, batch_first = True)
        self.linear1 = nn.Linear(self.n_hidden, self.n_hidden)
        self.linear2 = nn.Linear(self.n_hidden, self.n_hidden)
        self.flat = nn.Flatten()
        self.linear3 = nn.Linear(self.n_hidden, 2*self.n_latent)
        self.encoder = nn.Sequential(
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Flatten(),
            nn.Linear(self.n_hidden, 2*self.n_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.n_latent, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
            nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
        )

    def forward(self, x):        
        out, hidden = self.lstm(x)
        ## Encode
        x = self.encoder(out)
        mu, logvar = torch.split(x, [self.n_latent, self.n_latent], dim=-1)
        ## Reparameterization trick
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        x = mu + eps*std
        ## Decoder
        x = self.decoder(x)
        return x, mu, logvar, hidden
    
def train(model, optimizer, train_loader, loss_fn):
    model.train()
    batch_losses = [] 

    for batch in train_loader:
        batch_size, seq_len, _ = batch.shape
        batch = batch.view(batch_size, -1).float().to(device)
        
        y_pred, mu, logvar, hidden = model(batch)
        loss = loss_fn(batch, y_pred, mu, logvar)      

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
        
        batch_losses += [loss.item()]    

    return batch_losses

def val(model, val_loader, loss_fn):
    model.eval()
    val_losses = [] 
    
    for batch in val_loader: 
        batch_size, seq_len, _ = batch.shape
        batch = batch.view(batch_size, -1).float().to(device)
        
        y_hat, mu, logvar, hidden = model(batch)
        loss = loss_fn(batch, y_hat, mu, logvar)
        
        val_losses += [loss.item()]  
        
    return np.mean(val_losses)

def test(model, test_loader, loss_fn):      
    with torch.no_grad():
        labels = []
        scores = []        

        model.eval()  
        for batch in test_loader:
            batch_size, seq_len, _ = batch[0].shape
            x, y = batch[0].view(batch_size, -1).float().to(device), batch[1].float().to(device)
            y_hat, mu, logvar, hidden = model(x)
      
            loss = loss_fn(x, y_hat, mu, logvar)
        
            mse = nn.functional.mse_loss(x, y_hat, reduction='none').cpu()
            mse = mse.reshape(batch_size, seq_len, -1)
            mse = torch.mean(mse, 2).tolist()
            
            scores += mse                
            labels += y.cpu().tolist()
            
        return flatten(scores), flatten(labels)
    
def omni_experiment(train_loader, val_loader, test_loader, args):
    model_name = "omnianomaly"
    print("Evaluating " + model_name)

    hidden_dim = args['hidden_dim']
    if  hidden_dim == 'default':
        hidden_dim = 500        
    latent_dim =  12

    model = OmniAnomaly(args['input_dim']*args['seq_len'], hidden_dim, latent_dim).to(device)
    loss_fn = kl_recon_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)

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
    scores, labels = test(model, test_loader, loss_fn)

    thresh, auc = ROC(labels, scores, verbose = True)
    metrics_dict = {
    "model" : [model_name], 
    "metric" : ["auc-roc"],
    "score" : [auc]}

    return metrics_dict
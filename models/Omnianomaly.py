import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
from utils.utils import kl_recon_loss, flatten, ROC
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class OmniAnomaly(nn.Module):
    def __init__(self, feats, hidden, latent = 3):
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
            
        return scores, labels


# class PlanarFlow(nn.Module):
#     """Single planar normalizing flow: z' = z + u * tanh(w^T z + b)."""
#     def __init__(self, n_latent):
#         super(PlanarFlow, self).__init__()
#         self.w = nn.Parameter(torch.empty(n_latent).normal_(0, 0.01))
#         self.b = nn.Parameter(torch.zeros(1))
#         self.u = nn.Parameter(torch.empty(n_latent).normal_(0, 0.01))

#     def forward(self, z):
#         # z: (seq, batch, n_latent)
#         lin = (z * self.w).sum(-1, keepdim=True) + self.b          # (seq, batch, 1)
#         # Enforce invertibility: w^T u_hat >= -1
#         wu = (self.w * self.u).sum()
#         m_wu = -1.0 + F.softplus(wu)
#         u_hat = self.u + (m_wu - wu) * self.w / ((self.w * self.w).sum() + 1e-8)
#         z_new = z + u_hat * torch.tanh(lin)
#         psi = (1.0 - torch.tanh(lin) ** 2) * self.w                # (seq, batch, n_latent)
#         log_det = torch.log(torch.abs(1.0 + (psi * u_hat).sum(-1, keepdim=True)) + 1e-8)
#         return z_new, log_det                                        # log_det: (seq, batch, 1)


# class OmniAnomaly(nn.Module):
#     def __init__(self, feats, n_hidden, n_flows=20):
#         super(OmniAnomaly, self).__init__()
#         self.name = 'OmniAnomaly'
#         self.beta = 0.01
#         self.n_feats = feats
#         self.n_hidden = n_hidden
#         self.n_latent = 3
#         # Encoder RNN + MLP (outputs mu and log_std, not log_var)
#         self.encoder_rnn = nn.GRU(feats, self.n_hidden, 2)
#         self.encoder_mlp = nn.Sequential(
#             nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
#             nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
#             nn.Linear(self.n_hidden, 2 * self.n_latent)
#         )
#         # Planar normalizing flows on the latent sample
#         self.flows = nn.ModuleList([PlanarFlow(self.n_latent) for _ in range(n_flows)])
#         # Decoder RNN + MLP (no Sigmoid — data need not be in [0,1])
#         self.decoder_rnn = nn.GRU(self.n_latent, self.n_hidden, 2)
#         self.decoder_mlp = nn.Sequential(
#             nn.Linear(self.n_hidden, self.n_hidden), nn.PReLU(),
#             nn.Linear(self.n_hidden, self.n_feats)
#         )

#     def forward(self, x, hidden=None):
#         # x: (batch, win, feats)
#         bs, win, _ = x.shape
#         x_seq = x.view(win, bs, self.n_feats)           # (win, bs, feats)

#         # Encode
#         enc_out, hidden = self.encoder_rnn(x_seq, hidden)   # enc_out: (win, bs, n_hidden)
#         enc = self.encoder_mlp(enc_out)                      # (win, bs, 2*n_latent)
#         mu, log_std = torch.split(enc, self.n_latent, dim=-1)
#         std = torch.exp(log_std)
#         z = mu + torch.randn_like(std) * std                 # (win, bs, n_latent)

#         # Planar normalizing flows
#         log_det_sum = z.new_zeros(win, bs, 1)
#         for flow in self.flows:
#             z, log_det = flow(z)
#             log_det_sum = log_det_sum + log_det

#         # Decode (RNN then MLP — no Sigmoid)
#         dec_out, _ = self.decoder_rnn(z)                     # (win, bs, n_hidden)
#         x_hat = self.decoder_mlp(dec_out)                    # (win, bs, n_feats)

#         # Permute back to (bs, win, feats / n_latent) for downstream use
#         x_hat   = x_hat.permute(1, 0, 2)                    # (bs, win, feats)
#         mu      = mu.permute(1, 0, 2)                        # (bs, win, n_latent)
#         log_std = log_std.permute(1, 0, 2)                   # (bs, win, n_latent)
#         log_det_sum = log_det_sum.squeeze(-1).permute(1, 0)  # (bs, win)


#         return x_hat.reshape(bs, win * self.n_feats),mu.reshape(bs, win * self.n_latent),log_std.reshape(bs, win * self.n_latent), log_det_sum, hidden

    
# def train(model, optimizer, train_loader, criterion):
#     model.train()
#     batch_losses = [] 
#     for batch in train_loader: 
#         batch_size, seq_len, input_dim = batch.shape
#         batch = batch.float().to(device)    
        
#         # batch = batch.view(batch_size, -1).float().to(device)        
        
#         y_pred, mu, log_std, log_det_sum, hidden = model(batch, None)
#         hidden = hidden.detach()
#         d_flat = batch.view(-1, input_dim * seq_len)
#         # Reconstruction NLL: -log p(x|z) = 0.5 * sum((x - x_hat)^2)
#         recon = 0.5 * torch.sum(criterion(y_pred, d_flat), dim=-1)
#         # KL with normalizing-flow correction: KL[q(z0)||p(z)] - sum_k log|det J_k|
#         std = torch.exp(log_std)
#         kl_gauss = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 1.0 - 2.0 * log_std, dim=-1)
#         KLD = kl_gauss - log_det_sum.sum(dim=-1)
#         loss = torch.mean(recon + model.beta * KLD)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         batch_losses += [loss.item()]   
        
#     return batch_losses 

# def val(model, val_loader, criterion):
#     model.eval()
#     val_losses = []
    
#     for batch in val_loader: 
#         batch_size, seq_len, input_dim = batch.shape
#         batch = batch.float().to(device)    

#         y_pred, mu, log_std, log_det_sum, hidden = model(batch, None)
#         hidden = hidden.detach()
#         d_flat = batch.view(-1, input_dim * seq_len)
#         recon = 0.5 * torch.sum(criterion(y_pred, d_flat), dim=-1)
#         std = torch.exp(log_std)
#         kl_gauss = 0.5 * torch.sum(mu.pow(2) + std.pow(2) - 1.0 - 2.0 * log_std, dim=-1)
#         KLD = kl_gauss - log_det_sum.sum(dim=-1)
#         loss = torch.mean(recon + model.beta * KLD)
        
#         val_losses += [loss.item()]
        
#     return np.mean(val_losses)

# def test(model, test_loader, criterion):
#     with torch.no_grad():
#         model.eval()
        
#         labels = []
#         scores = []
        
#         for batch in test_loader: 
#             batch_size, seq_len, input_dim = batch[0].shape
#             x, y = batch[0].float().to(device), batch[1].float().to(device)
#             y_pred, mu, log_std, log_det_sum, hidden = model(x, None)
#             hidden = hidden.detach()

#             # Score = negative ELBO on the last time-step
#             # y_pred_last = y_pred.view(batch_size, seq_len, input_dim)[:, -1, :]  # (bs, feats)
#             # d_last      = x[:, -1, :]                                            # (bs, feats)
            
#             d_flat = x.view(-1, input_dim * seq_len)

#             mse = criterion(y_pred, d_flat)
#             mse = mse.reshape(batch_size, seq_len, -1)            
#             recon_last = 0.5 * torch.sum(mse, dim=-1)
            
#             std = torch.exp(log_std)
#             t = mu.pow(2) + std.pow(2) - 1.0 - 2.0 * log_std
#             t = t.reshape(batch_size, seq_len, -1) 
#             kl_gauss = 0.5 * torch.sum(t, dim=-1)     
            
#             KLD = kl_gauss - log_det_sum
#             score = recon_last + model.beta * KLD    


#             scores += score.cpu().tolist()
#             labels += y.cpu().tolist()
            
#         return labels, scores
    
def omni_experiment(train_loader, val_loader, test_loader, args):
    model_name = "omnianomaly"
    print("Evaluating " + model_name)

    hidden_dim = args['hidden_dim']
    if  hidden_dim == 'default':
        hidden_dim = 500        

    model = OmniAnomaly(args['input_dim']*args['seq_len'], hidden_dim).to(device)
    # loss_fn = nn.MSELoss(reduction = 'none')
    loss_fn = kl_recon_loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-5)

    best_val = 10000

    for i in range(args['num_epochs']):
        train(model, optimizer, train_loader, loss_fn)
        val_losses = val(model, val_loader, loss_fn)
        
        if args['verbose']:
            print('Epoch %d Val Loss: %f' % (i,val_losses))
        if np.isnan(val_losses):
            break 
        if val_losses < best_val:
            best_val = val_losses 
            best_model_state_dict = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), args['experiment_dir'] + "/" + model_name + ".pth")

    model.load_state_dict(torch.load(args['experiment_dir'] + "/" + model_name + ".pth"))
    scores, labels = test(model, test_loader, loss_fn)

    return flatten(labels), flatten(scores)

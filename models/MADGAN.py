import torch
import torch.nn as nn
import copy
import numpy as np
from utils.utils import kl_recon_loss, flatten, ROC
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


default_dim = {
'swat': 100,
'wadi': 100, 
'smap': 64,
'msl': 64,
'smd': 100,
'ucr': 64,
}

class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, input_dim, n_lstm_layers = 3):
        super().__init__()
        self.hidden_units = hidden_dim
        self.n_lstm_layers = n_lstm_layers
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.lstm = nn.LSTM(input_size=self.latent_dim,
                            hidden_size=self.hidden_units,
                            num_layers=self.n_lstm_layers,
                            batch_first=True,
                            dropout=.1)

        self.linear = nn.Linear(in_features=self.hidden_units,
                                out_features=self.input_dim)

    def forward(self, x):
        rnn_output, _ = self.lstm(x)
        return self.linear(rnn_output)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_units, n_lstm_layers = 1, add_batch_mean = False):
        super().__init__()
        
        self.hidden_units = hidden_units
        self.input_dim = input_dim
        self.n_lstm_layers = n_lstm_layers

        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=self.hidden_units,
                            num_layers=self.n_lstm_layers,
                            batch_first=True)
        
        self.linear = nn.Linear(in_features=self.hidden_units,
                                out_features=1)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        rnn_output, _ = self.lstm(x)
        return self.activation(self.linear(rnn_output))
    
    
def train(generator, discriminator, latent_dim, optimizer_d, optimizer_g, 
          train_loader, loss_fn, D_rounds, G_rounds):       
    disc_losses = []
    gen_losses = []
    
    generator.train()
    discriminator.train() 
    for real in train_loader: 
        real = real.float().to(device)
        
        batch_size, seq_len = real.shape[0], real.shape[1]
        real_labels = torch.zeros(batch_size, seq_len, 1, device=device) 
        fake_labels = torch.ones(batch_size, seq_len, 1, device=device) 
        
        #train discriminator 
        for i in range(D_rounds):
            discriminator.zero_grad()
            
            #real data            
            real_pred = discriminator(real)
            real_loss = loss_fn(real_pred, real_labels)
            real_loss.backward() 

            #fake data
            rand = torch.randn(batch_size, seq_len, latent_dim, device=device).float()
            fakes = generator(rand)
            fakes_pred = discriminator(fakes)
            fakes_loss = loss_fn(fakes_pred, fake_labels) 
            fakes_loss.backward()            
            optimizer_d.step()    
            
        disc_losses.append(fakes_loss.item() + real_loss.item())
        
        #train generator 
        for i in range(G_rounds):
            generator.zero_grad()     
            
            rand = torch.randn(batch_size, seq_len, latent_dim, device=device).float()             
            fakes = generator(rand)             
            fakes_pred = discriminator(fakes)
            fakes_loss = loss_fn(fakes_pred, fake_labels) 
            fakes_loss.backward() 
            optimizer_g.step()
        
        gen_losses.append(fakes_loss.item())    
        
    return np.mean(disc_losses), np.mean(gen_losses) 
            
def val(generator, discriminator, latent_dim, val_loader, loss_fn):   
    disc_losses = [] 
    gen_losses = []
    
    generator.eval()
    discriminator.eval() 
    for real in val_loader:    
        real = real.float().to(device)
        
        batch_size, seq_len = real.shape[0], real.shape[1]
        real_labels = torch.zeros(batch_size, seq_len, 1, device=device) 
        fake_labels = torch.ones(batch_size, seq_len, 1, device=device) 
        
        #eval discriminator             
        #real data            
        real_pred = discriminator(real)
        real_loss = loss_fn(real_pred, real_labels)

        #fake data
        rand = torch.randn(batch_size, seq_len, latent_dim, device=device).float()   
        fakes = generator(rand)
        fakes_pred = discriminator(fakes)
        fakes_loss = loss_fn(fakes_pred, fake_labels) 

        disc_losses.append(fakes_loss.item() + real_loss.item())
        
        #eval generator 
        rand = torch.randn(batch_size, seq_len, latent_dim, device=device).float()                  
        fakes = generator(rand)             
        fakes_pred = discriminator(fakes)
        fakes_loss = loss_fn(fakes_pred, fake_labels)         
        gen_losses.append(fakes_loss.item())    
        
    return np.mean(disc_losses), np.mean(gen_losses) 

def covariance_similarity(tensor1, tensor2):
    mean1 = tensor1.mean(dim=-1)
    mean1_broadcasted = torch.broadcast_tensors(tensor1.T, mean1.T)[1].T
    tensor1_center = tensor1 - mean1_broadcasted
    
    if tensor1.shape[2] == 1:
        std1 = tensor1.std(dim=2, correction = 0)
    else: 
        std1 = tensor1.std(dim=2)        
    std1_broadcasted = torch.broadcast_tensors(tensor1.T, std1.T)[1].T
    
    
    mean2 = tensor2.mean(dim=-1)
    mean2_broadcasted = torch.broadcast_tensors(tensor2.T, mean2.T)[1].T
    tensor2_center = tensor2 - mean2_broadcasted   
    
    if tensor2.shape[2] == 1:
        std2 = tensor2.std(dim=2, correction = 0)
    else: 
        std2 = tensor2.std(dim=2)
    std2_broadcasted = torch.broadcast_tensors(tensor2.T, std2.T)[1].T
    
    std_broadcasted = std1_broadcasted * std2_broadcasted
    res = tensor1_center * tensor2_center / (std_broadcasted + 1e-7)
    res = res.mean(axis=2)

    return res

def reconstruction_loss(data, generator, latent_dim, tolerance = 1e-3, max_iter = 70):
    generator.train()
    rand = torch.randn(data.shape[0], data.shape[1], latent_dim, requires_grad = True, device=device)
    optimizer_r = torch.optim.RMSprop([rand])    
    
    prev_loss = 100000    
    for i in range(max_iter):        
        generated = generator(rand)    
        
        reconstruction_loss = 1 - covariance_similarity(data, generated)    
        reconstruction_loss.mean().backward()
        optimizer_r.step()
        rand.grad.zero_()
        
        current_loss = reconstruction_loss.max().detach().cpu().numpy()
        if np.abs(prev_loss - current_loss) < tolerance:
            break        
        prev_loss = current_loss
            
    return current_loss 
            
def test(generator, discriminator, latent_dim, test_loader, lmbda = 0.5):
    labels = []
    anomaly_scores = []
    
    for batch in test_loader: 
        x, y = batch[0].float().to(device), batch[1].float().to(device)

        r_loss = reconstruction_loss(x, generator, latent_dim)
        pred = discriminator(x)
        
        anomaly_score = lmbda*r_loss + (1-lmbda) * pred
        
        labels += [y]  
        anomaly_scores += [anomaly_score.squeeze(-1)]
        
    labels = torch.cat(labels, axis = 0).tolist()
    anomaly_scores = torch.cat(anomaly_scores, axis = 0).tolist()    

    return flatten(labels),flatten(anomaly_scores)
    
def madgan_experiment(train_loader, val_loader, test_loader, args):
    model_name = "madgan"
    print("Evaluating " + model_name)
    
    loss_fn = nn.BCELoss()

    hidden_dim = args['hidden_dim']
    if  hidden_dim == 'default':
        hidden_dim = default_dim[args['dataset']]
        
    latent_dim = 12

    generator = Generator(latent_dim, hidden_dim, args['input_dim'])
    discriminator = Discriminator(args['input_dim'], hidden_dim) 

    generator = generator.to(device)
    discriminator = discriminator.to(device)
    optimizer_d = torch.optim.Adam(discriminator.parameters())
    optimizer_g = torch.optim.Adam(generator.parameters())

    best_val = 10000
    for i in range(args['num_epochs']):
        losses = train(generator, discriminator, latent_dim, optimizer_d, optimizer_g, 
          train_loader, loss_fn, 1, 3)
        
        d_losses, g_losses = val(generator, discriminator, latent_dim, val_loader, loss_fn)

        if d_losses < best_val and best_val - d_losses > 1e-5:
            best_val = d_losses 
            torch.save({"generator" : generator.state_dict(), "discriminator" : discriminator.state_dict()}, args['experiment_dir'] + "/" + model_name + ".pth")

        if(args['verbose']):
            print('Epoch {:d} Val Loss: {:f}'.format(i,d_losses))

    checkpoint = torch.load(args['experiment_dir'] + "/" + "madgan" + ".pth", weights_only=True)

    generator.load_state_dict(checkpoint["generator"])
    discriminator.load_state_dict(checkpoint["discriminator"])

    labels, scores = test(generator, discriminator, latent_dim, test_loader)
    thresh, auc = ROC(labels, scores)

    metrics_dict = {
    "model" : [model_name], 
    "metric" : ["auc-roc"],
    "score" : [auc]}

    return metrics_dict

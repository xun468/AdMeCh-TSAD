import torch
import torch.nn as nn
import copy
from utils.utils import flatten, ROC
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

default_dim = {
'swat': 40,
'wadi': 100, 
'smap': 55,
'msl': 33,
'smd': 38,
'ucr': 55,
}

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class Encoder(nn.Module):
  def __init__(self, in_size, latent_size):
    super().__init__()
    self.linear1 = nn.Linear(in_size, int(in_size/2))
    self.linear2 = nn.Linear(int(in_size/2), int(in_size/4))
    self.linear3 = nn.Linear(int(in_size/4), latent_size)
    self.relu = nn.ReLU(True)
        
  def forward(self, w):
    out = self.linear1(w)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    z = self.relu(out)
    return z
    
class Decoder(nn.Module):
  def __init__(self, latent_size, out_size):
    super().__init__()
    self.linear1 = nn.Linear(latent_size, int(out_size/4))
    self.linear2 = nn.Linear(int(out_size/4), int(out_size/2))
    self.linear3 = nn.Linear(int(out_size/2), out_size)
    self.relu = nn.ReLU(True)
    self.sigmoid = nn.Sigmoid()
        
  def forward(self, z):
    out = self.linear1(z)
    out = self.relu(out)
    out = self.linear2(out)
    out = self.relu(out)
    out = self.linear3(out)
    w = self.sigmoid(out)
    return w
    
class UsadModel(nn.Module):
  def __init__(self, w_size, z_size, seq_len, input_dim):
    super().__init__()
    self.encoder = Encoder(w_size, z_size)
    self.decoder1 = Decoder(z_size, w_size)
    self.decoder2 = Decoder(z_size, w_size)
    
    self.seq_len = seq_len
    self.input_dim = input_dim
  
  def training_step(self, batch, n):
    z = self.encoder(batch)
    w1 = self.decoder1(z)
    w2 = self.decoder2(z)
    w3 = self.decoder2(self.encoder(w1))
        
    loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
    loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return loss1,loss2

  def validation_step(self, batch, n):
    with torch.no_grad():
        z = self.encoder(batch)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        loss1 = 1/n*torch.mean((batch-w1)**2)+(1-1/n)*torch.mean((batch-w3)**2)
        loss2 = 1/n*torch.mean((batch-w2)**2)-(1-1/n)*torch.mean((batch-w3)**2)
    return {'val_loss1': loss1, 'val_loss2': loss2}
        
  def validation_epoch_end(self, outputs):
    batch_losses1 = [x['val_loss1'] for x in outputs]
    epoch_loss1 = torch.stack(batch_losses1).mean()
    batch_losses2 = [x['val_loss2'] for x in outputs]
    epoch_loss2 = torch.stack(batch_losses2).mean()
    return {'val_loss1': epoch_loss1.item(), 'val_loss2': epoch_loss2.item()}
    
  def epoch_end(self, epoch, result):
    print("Epoch [{}], val_loss1: {:.4f}, val_loss2: {:.4f}".format(epoch, result['val_loss1'], result['val_loss2']))
    
def training(model, train_loader, val_loader, args):
    history = []
    optimizer1 = torch.optim.Adam(list(model.encoder.parameters())+list(model.decoder1.parameters()))
    optimizer2 = torch.optim.Adam(list(model.encoder.parameters())+list(model.decoder2.parameters()))
    best_val = 10000
    
    for epoch in range(args['num_epochs']):
        for batch in train_loader:
            batch=to_device(batch,device)
            b = batch.float().flatten(1)

            #Train AE1
            loss1,loss2 = model.training_step(b,epoch+1)
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            
            
            #Train AE2
            loss1,loss2 = model.training_step(b,epoch+1)
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()
            
        result = evaluate(model, val_loader, epoch+1)
        if result["val_loss2"] < best_val: 
            best_val = result["val_loss2"]
            best_model_state_dict = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), args['experiment_dir'] + "/usad.pth")

        
        if args['verbose']:
            model.epoch_end(epoch, result)
        history.append(result)
    return history

def testing(model, test_loader, seq_len, alpha=.5, beta=.5):
    scores=[]
    labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch=to_device(batch,device)
            b = batch[0].float().flatten(1)
            y = batch[1]
        
            w1=model.decoder1(model.encoder(b))
            w2=model.decoder2(model.encoder(w1))             

            term1 = torch.reshape((b-w1)**2, (b.shape[0], seq_len, -1))
            term2 = torch.reshape((b-w2)**2, (b.shape[0], seq_len, -1))
            scores += (alpha*torch.mean(term1,2) + beta*torch.mean(term2,2)).cpu().tolist()
                
            labels += y.cpu().tolist()    

    return flatten(scores), flatten(labels)

def evaluate(model, val_loader, n):
    outputs = [model.validation_step(to_device(batch.float().flatten(1),device), n) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def usad_experiment(train_loader, val_loader, test_loader, args):
    model_name = "usad"
    print("Evaluating " + model_name)
    
    hidden_dim = args['hidden_dim']
    if  hidden_dim == 'default':
        hidden_dim = default_dim[args['dataset']]

    w_size=args['seq_len']*args['input_dim']
    z_size=args['seq_len']*hidden_dim

    print(w_size)
    print(z_size)

    model = UsadModel(w_size, z_size, args['seq_len'], args['input_dim'])
    model = to_device(model,device)

    history = training(model, train_loader, val_loader, args)

    best_model_state_dict = torch.load(args['experiment_dir'] + "/" + model_name + ".pth")
    model.load_state_dict(best_model_state_dict)

    scores, labels = testing(model, test_loader, args['seq_len'])

    thresh, auc = ROC(labels, scores)
    metrics_dict = {
    "model" : [model_name], 
    "metric" : ["auc-roc"],
    "score" : [auc]}
    
    return metrics_dict
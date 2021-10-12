from __future__ import print_function
import pandas as pd
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,MultiStepLR
import time
import math
from sklearn.metrics import mean_absolute_error
from pickle import load

from loss.dilate_loss import dilate_loss
alpha=0.095
gamma = 0.09

train_dat = pd.read_csv('new_data/new_train.csv')
val_dat = pd.read_csv('new_data/new_val.csv')
test_dat = pd.read_csv('new_data/new_test.csv')
feature_names = ['pm25','pm10','um1','um03','um05','ae25','ae10']
target_names =['pm25_station']

train_features = train_dat[feature_names].values.astype(np.float32)
train_target = train_dat[target_names].values.astype(np.float32)
val_features = val_dat[feature_names].values.astype(np.float32)
val_target = val_dat[target_names].values.astype(np.float32)
test_features = test_dat[feature_names].values.astype(np.float32)
test_target = test_dat[target_names].values.astype(np.float32)

scaler_X = load(open('new_data/scaler_X.pkl','rb'))
scaler_y = load(open('new_data/scaler_y.pkl','rb'))



class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, y, seq_len=1):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        #return (self.X[index:index+self.seq_len], self.y[index+self.seq_len-1])
        return (self.X[index:index+self.seq_len], self.y[index:index+self.seq_len])
    

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


    
class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()


    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)


    def forward(self, x, mask=None, return_attention=True):
        batch_size, seq_length, embed_dim = x.size()
        
        #print('\n debug....')
        #print(self.input_dim,self.embed_dim,self.num_heads,self.head_dim)
        #print(x.shape)
        #time.sleep(100)
        
        qkv = self.qkv_proj(x)
        #print(qkv.shape)
        #time.sleep(100)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        print('debug...q,k,v')
        print(q.shape,k.shape,v.shape)
        
        values, attention = scaled_dot_product(q, k, v, mask=mask)

        print(values.shape,attention.shape)
        time.sleep(100)

        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        #o = values

        if return_attention:
            return o, attention
        else:
            return o

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=12):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()

        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        #print('budeg...')
        
        n = x.shape[0]
        tmp = self.pe[:, :x.size(1)].repeat(n,1, 1)
        x = torch.cat((x,tmp),dim=2)

        #x = x + self.pe[:, :x.size(1)]

        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.self_attn = MultiheadAttention(18*2,18*2,1)
        self.fc = nn.Linear(9+9,1)
        self.fc1 = nn.Linear(36,1)
        self.fc2= nn.Linear(18,18)
        self.fc3 = nn.Linear(2,1)
        
        self.input_fc = nn.Linear(7,18)

        # Layers to apply in between the main layers
        self.norm = nn.LayerNorm(18)
        self.norm1 = nn.LayerNorm(18)
        self.norm2 = nn.LayerNorm(18)
        self.dropout = nn.Dropout(0.1)
        self.m = nn.BatchNorm1d(12, affine=False)
        
        self.positional_encoding = PositionalEncoding(d_model=18)


         # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(18, 9),
            nn.ReLU(inplace=True),
            nn.Linear(9, 1)
        )

    def forward(self, x):

        x  = self.input_fc(x)
        x = self.positional_encoding(x)

        x_old = x
        

        attn_out,attn= self.self_attn(x,mask=None)

        x = x_old + (attn_out)

        
        x = self.m(x)


        x = self.fc1(x)
        

        output = x
        return output,attn
   
    
    
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)
        
        output,attn = model(data)
        #loss = F.mse_loss(output, target)
        loss_l1 = nn.L1Loss()(output,target)

      

        device = torch.device("cuda")
        loss, loss_shape, loss_temporal = dilate_loss(target,output,alpha, gamma, device)
        
        loss = loss + 0.15*loss_l1
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

    
    

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    all_results = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
        #for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output,attn = model(data)

            np.save('attention_map/n_'+str(batch_idx)+'.npy',attn[0,0,::].cpu().numpy())

            
            all_results.append(output[0,:,:].cpu().view(-1).tolist())

            tmp = nn.L1Loss()(output, target)

                
            test_loss = test_loss + tmp

    print(test_loss)
    print(len(test_loader.dataset))
            
    test_loss /= len(test_loader.dataset)
    
    
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return all_results
    
    
    
    
def all_pred(pred,seq_len=12):
    
    all_results = []
    for i in range(0,len(pred)):
        tmp = np.empty(seq_len)
        tmp[:] = np.NaN

        
        
        ## fill the tmp
        begin = 0 if (i+1-seq_len)<=0 else (i+1-seq_len)
        for j in range(begin,i+1):
            #print(j,j-begin,j,i-j)
            tmp[j-begin] = pred[j,i-j]
        
        all_results.append(tmp)
            
        
    all_results_df  = pd.DataFrame(all_results)
    
    return all_results_df
    
    
    
    
    
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (defaut: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    
    
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': False}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)



    train_dataset = TimeseriesDataset(train_features, train_target, seq_len=12)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 8, shuffle = False)
    val_dataset = TimeseriesDataset(val_features, val_target, seq_len=12)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1, shuffle = False)   
    test_dataset = TimeseriesDataset(test_features, test_target, seq_len=12)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)
    


    model = Net().to(device)
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=0.000359)


    all_test = []
    all_test_old = []
    
    best_val = 100
    best_inverse = 100 
    best_epoch = 0



########################start of training#################  

    for epoch in range(1, args.epochs + 1):
        #scheduler.step()

        train(args, model, device, train_loader, optimizer, epoch)
        
       ### get the validation MAE and save the current best model!
        pred = test(model, device, val_loader)
        pred = np.array(pred)
        pred_df = all_pred(pred,12)
        pred_df = pred_df.apply(lambda row: row.fillna(row.mean()), axis=1)
        pred_df['mean'] = pred_df.iloc[:, 0:23].mean(axis=1)
        
        pred_true = val_target[:pred_df.shape[0],0]
        pred_df['true'] = pred_true
        val_mae = (mean_absolute_error(pred_true,pred_df['mean']))
        
        inverse_pred = scaler_y.inverse_transform(pred_df['mean'].values.reshape(-1,1))
        inverse_true = scaler_y.inverse_transform(pred_true.reshape(-1,1))
        inverse_val = mean_absolute_error(inverse_true,inverse_pred)
        
        print('val mae: ', val_mae)
        if val_mae < best_val:
            print('renew best model and save...',val_mae)
            best_val = val_mae
            best_inverse = inverse_val
            best_epoch = epoch
            
            pred_df.to_csv('result/tmp/best_val.csv',index=False)
            torch.save(model, "result/tmp/model.pt")   



########################end of training#################   
        

        

    ## now use the found best model to predict the test data
    best_model = torch.load('result/tmp/model.pt')
    pred = test(best_model, device, test_loader)
    pred = np.array(pred)
    pred_df = all_pred(pred,12)
    pred_df = pred_df.apply(lambda row: row.fillna(row.mean()), axis=1)
    pred_df['mean'] = pred_df.iloc[:, 0:11].mean(axis=1)
        
    pred_true = test_target[:pred_df.shape[0],0]
    pred_df['true'] = pred_true
    
    inverse_pred = scaler_y.inverse_transform(pred_df['mean'].values.reshape(-1,1))
    inverse_true = scaler_y.inverse_transform(pred_true.reshape(-1,1))
    
    
    pred_df.to_csv('result/tmp/test.csv',index=False)
    
    print('Best Val MAE: ', best_val)
    print('Best Val MAE: ', best_inverse)
    print('Best Val Epoch: ', best_epoch)
    print('Test MAE: ',mean_absolute_error(pred_true,pred_df['mean']))
    print('Test MAE: ',mean_absolute_error(inverse_true,inverse_pred))
    


if __name__ == '__main__':
    main()

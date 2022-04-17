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
from ptflops import get_model_complexity_info


########
#model transfer
import onnx
from onnx_tf.backend import prepare
from torch.autograd import Variable

########


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


# https://stackoverflow.com/questions/57893415/pytorch-dataloader-for-time-series-task

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


    def forward(self, x, mask=None, return_attention=False):
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
        values, attention = scaled_dot_product(q, k, v, mask=mask)
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
        #print(x.shape)
        #print(self.pe[:, :x.size(1)].shape)
        
        n = x.shape[0]
        
        tmp = self.pe[:, :x.size(1)].repeat(n,1, 1)

        #print('debug...')
        #print(x.shape,tmp.shape)

        x = torch.cat((x,tmp),dim=2)
        #print(x.shape)
        #time.sleep(100)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.self_attn = MultiheadAttention(18*2,18*2,1)
        self.fc = nn.Linear(9+9,1)
        self.fc1 = nn.Linear(36,1)
        self.fc2= nn.Linear(9,1)
        self.fc3 = nn.Linear(2,1)
        
        self.input_fc = nn.Linear(7,18)

        # Layers to apply in between the main layers
        self.norm = nn.LayerNorm(18)
        self.norm1 = nn.LayerNorm(9)
        self.norm2 = nn.LayerNorm(9)
        self.dropout = nn.Dropout(0.1)
        self.m = nn.BatchNorm1d(12, affine=False)
        
        self.positional_encoding = PositionalEncoding(d_model=18)


         # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(18, 9),
            nn.ReLU(inplace=True),
            nn.Linear(9, 1)
        )

        ### used for soft project
        # create temperature variable
        self._temperature = torch.nn.Parameter(
            torch.tensor(
                1.0,
                requires_grad=True,
                dtype=torch.float32,
            )
        )

        self._min_sigma = torch.tensor(1e-4, dtype=torch.float32)


    def step_mean(self,x,n):
        #print('debug....')
        x = torch.cat(([torch.mean(x[:,i:(i+n),:],keepdim=True,dim=1)for i in range(0,x.shape[1],n)]),dim=1)
        return x
    
    def step_softmax(self,dist,n):
        #print('debug...')
        weights = torch.cat(([torch.softmax(dist[:,i:(i+n),:],dim=1)for i in range(0,dist.shape[1],n)]),dim=1)
        return weights
    
    def project_x(self,x,weights,n):
        #print('debug...')

        weighted_x = x * weights
        project_x = torch.cat(([torch.mean(weighted_x[:,i:(i+n),:],keepdim=True,dim=1)for i in range(0,weighted_x.shape[1],n)]),dim=1)
        return project_x


    def sigma(self):
        device = self._temperature.device
        return torch.max(self._temperature ** 2, self._min_sigma.to(device))

    def _get_distances(self, all_points, query_points,n):
        # print('debug dist...')
        deltas = all_points - query_points.repeat(1,n,1) + 1e-6
        dist = torch.sum(deltas ** 2, dim=2, keepdim=True) / self.sigma()
        return dist

    def forward(self, x):

        x  = self.input_fc(x)
        x = self.positional_encoding(x)
        x_old = x

        ##### use mean as representative points
        #x = self.step_mean(x,3)

        #### soft project
        SR = 1
        query_points = self.step_mean(x,SR)
        dist = self._get_distances(x,query_points,SR)
        weights = self.step_softmax(dist,SR)
        project_x = self.project_x(x,weights,SR)

        
        attn_out = self.self_attn(project_x,mask=None)
        x = project_x + attn_out 

        attn_out = self.self_attn(x,mask=None)



        attn_out = attn_out.repeat(1,SR,1)
        x = x_old + (attn_out)

        x = self.m(x)

        x = (self.fc1(x))
        
        #time.sleep(100)
        ##x = self.linear_net(x)
        #x = x + self.dropout(linear_out)
        #x = self.norm2(x)
        # print(x_new.shape)
        # print(x_new[0,0,:])
        # print(x[0,0,:])
        #x = x + x_new
        #print(x[0,0,:])
        #x = torch.cat((x[:,:,4:5],x_new),dim=2)
        #x = torch.cat((x,x_new),dim=2)
        #print(x.shape)
        #x = self.fc(x)
        #print(x.shape)
        output = x
        #print(x.shape)
        #time.sleep(100)
        return output
   
    
    
    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # print('debug...')
        # print(type(data))
        # print(model.sigma())
        # time.sleep(100)
        data, target = data.to(device), target.to(device)
        # print(data.shape)
        # time.sleep(100)
        output = model(data)
        #loss = F.mse_loss(output, target)
        loss_l1 = nn.L1Loss()(output,target)

      

        device = torch.device("cuda")
        loss, loss_shape, loss_temporal = dilate_loss(target,output,alpha, gamma, device)
        

        l2 = model.sigma()
        loss = loss + 0.15*loss_l1 + 0.5*l2
        #print('\n debug...')
        #print(output.shape,target.shape)
        #print(loss)
        #time.sleep(100)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        #time.sleep(1)
    
    

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    
    all_results = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
        #for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # print('debug time...')
            # t = time.time()
            output = model(data)
            # print(time.time()-t)*1000*1600
            # time.sleep(100)

            #print('debug...')
            #torch.onnx.export(model, data, "tf/sf.onnx",opset_version=10)
            #print(data.shape,target.shape,output.shape)
            #print(target.cpu().view(-1))
            #time.sleep(200)
            
            all_results.append(output[0,:,:].cpu().view(-1).tolist())
            
            #if batch_idx == 1:
            #    print('\n debug...')
            #    print(data.shape,target.shape,output.shape)
            #    print(target[0,:,:].cpu().view(-1))
            #    print(output[0,:,:].cpu().view(-1))
            #    time.sleep(2)
            #test_loss += F.mse_loss(output, target, reduction='mean')
            #tmp = F.mse_loss(output, target, reduction='mean')
            tmp = nn.L1Loss()(output, target)
            #print(tmp)
            #if np.isnan(tmp.cpu()):
            #    print('here...')
            #    print(target[0,:,:].view(-1),output[0,:,:].view(-1))
                
                
            test_loss = test_loss + tmp
            #print(test_loss)
            #time.sleep(1)
            
            #test_loss += F.mse_loss(output, target, reduction='sum').item()  # sum up batch loss
            #pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()

    print(test_loss)
    print(len(test_loader.dataset))
            
    test_loss /= len(test_loader.dataset)
    
    
    #print(len(all_results))
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return all_results
    
    
    
    
def all_pred(pred,seq_len=12):
    
    all_results = []
    for i in range(0,len(pred)):
        tmp = np.empty(seq_len)
        tmp[:] = np.NaN
        #print(tmp)
        
        
        ## fill the tmp
        begin = 0 if (i+1-seq_len)<=0 else (i+1-seq_len)
        for j in range(begin,i+1):
            #print(j,j-begin,j,i-j)
            tmp[j-begin] = pred[j,i-j]
        
        all_results.append(tmp)
        #print(tmp)
        #print('\n')
            
        
    all_results_df  = pd.DataFrame(all_results)
    
    return all_results_df
    
    
    
    
    
    
def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
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

#    transform=transforms.Compose([
#        transforms.ToTensor(),
#        transforms.Normalize((0.1307,), (0.3081,))
#        ])
#    dataset1 = datasets.MNIST('../data', train=True, download=True,
#                       transform=transform)
#    dataset2 = datasets.MNIST('../data', train=False,
#                       transform=transform)
#    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
#    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    train_dataset = TimeseriesDataset(train_features, train_target, seq_len=12)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 8, shuffle = False)
    val_dataset = TimeseriesDataset(val_features, val_target, seq_len=12)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 1, shuffle = False)   
    test_dataset = TimeseriesDataset(test_features, test_target, seq_len=12)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False)
    
    #print(test_target.shape)
    #print(len(test_loader))
    #time.sleep(100)

    model = Net().to(device)
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=0.000359)

    #scheduler = StepLR(optimizer, step_size=5, gamma=args.gamma)
    #scheduler = StepLR(optimizer, step_size=5, gamma=0.9)

    all_test = []
    all_test_old = []
    
    best_val = 100
    best_inverse = 100 
    best_epoch = 0
    
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
        
    #     pred = test(model, device, test_loader)
        
    #     pred = np.array(pred)
    #     #print(pred.shape)
    #     #time.sleep(100)


    #     pred_df = all_pred(pred,12)
        
    #     #print(pred_df.head(2))
    #     #print(pred_df.shape)
    #     pred_df = pred_df.apply(lambda row: row.fillna(row.mean()), axis=1)
    #     pred_df['mean'] = pred_df.iloc[:, 0:11].mean(axis=1)
        
        
    #     ## compute test MAE
    #     pred_true = test_target[:pred_df.shape[0],0]
    #     print(mean_absolute_error(pred_true,pred_df['mean']))
    #     all_test.append((mean_absolute_error(pred_true,pred_df['mean'])))
        
    #     pred_df['true'] = pred_true
    #     pred_df.to_csv('result/pred_'+str(epoch)+'.csv',index=False)


    #     ### inverse to original value
    #     inverse_pred = scaler_y.inverse_transform(pred_df['mean'].values.reshape(-1,1))
    #     inverse_true = scaler_y.inverse_transform(pred_true.reshape(-1,1))
    #     all_test_old.append(mean_absolute_error(inverse_true,inverse_pred))
    #     #time.sleep(2)
        
    #     #scheduler.step()
    # print(np.array(all_test))
    # print(np.array(all_test).min())
    # print(np.array(all_test_old).min())
    # print(np.argmin(np.array(all_test_old))+1)
    # if args.save_model:
    #     torch.save(model.state_dict(), "test.pt")
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

    macs, params = get_model_complexity_info(best_model, (12, 7), as_strings=False,
                                           print_per_layer_stat=True, verbose=True)

    print(macs,params)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))



    ### transfer to tf onnx
    torch.save(best_model.state_dict(), 'tf/sf.pth')

    # model = Net().to(device)
    dummy_input = Variable(torch.rand(1, 12, 7)).to(device)
    # torch.onnx.export(model, dummy_input, "tf/sf.onnx")
    torch.onnx.export(best_model, dummy_input, "tf/sf.onnx",opset_version=10)

    new_model = onnx.load('tf/sf.onnx')
    tf_rep = prepare(new_model) 
    tf_rep.export_graph('tf/sf.pb')

    # tf_rep.save('tf/sf.h5')
    # print('all ended...')

if __name__ == '__main__':
    main()

import torch
from torch import nn, cuda, backends, FloatTensor, LongTensor, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision.models import resnet34

model_meta = {
    resnet34:[8,6]
}

f_model = resnet34
sz = 224
bs = 64
drop = 0.4

cats = {
    1: 'ground',
    2: 'coconut_tree'
}

id2cat = list(cats.values())

cut,lr_cut = model_meta[f_model]

def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())

def split_by_idxs(seq, idxs):
    last = 0
    for idx in idxs:
        yield seq[last:idx]
        last = idx
    yield seq[last:]
    
def cut_model(m, cut):
    return list(m.children())[:cut] if cut else [m]

def get_base():
    layers = cut_model(f_model(True), cut)
    return nn.Sequential(*layers)

class StdConv(nn.Module):
    def __init__(self, n_in,n_out,stride=2,dp = 0.1):
        super().__init__()
        self.conv = nn.Conv2d(n_in,n_out,3,stride=stride,padding=1)
        self.bn = nn.BatchNorm2d(n_out)
        self.dropout = nn.Dropout(dp)
        
    def forward(self,x):
        return self.dropout(self.bn(F.relu(self.conv(x))))
    
def flatten_conv(x,k):
    bs,nf,gx,gy = x.size()
    x = x.permute(0,3,2,1).contiguous()
    return x.view(bs,-1,nf//k) 

class OutConv(nn.Module):
    def __init__(self, k, n_in, bias):
        super().__init__()
        self.k = k
        self.oconv1 = nn.Conv2d(n_in, (len(id2cat)+1) * k, 3, padding=1)
        self.oconv2 = nn.Conv2d(n_in, 4 * k, 3, padding = 1)
        self.oconv1.bias.data.zero_().add_(bias)
        
    def forward(self,x):
        return [flatten_conv(self.oconv1(x), self.k),
                flatten_conv(self.oconv2(x), self.k)]

class SaveFeatures():
    features=None
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = output
    def remove(self): self.hook.remove()

def get_base():
    layers = cut_model(f_model(True), cut)
    return nn.Sequential(*layers)


class SSD_Custom_noFPN1(nn.Module):
    def __init__(self, m_base, k, bias, drop):
        super().__init__()
        self.m_base = m_base
        
        # bottom up 
        self.sfs = [SaveFeatures(m_base[i]) for i in [5,6]] # 28x28 & 14x14
        self.drop = nn.Dropout(drop)
        self.sconv1 = StdConv(512,256, dp=drop, stride=1) # 7x7
        self.sconv2 = StdConv(256,256, dp=drop) # 4x4
        self.sconv3 = StdConv(256,256, dp=drop) # 2x2
        self.sconv4 = StdConv(256,256, dp=drop) # 1x1
                  
        # lateral
        self.lat1 = nn.Conv2d(128,256, kernel_size=1, stride=1, padding=0)

        # outconvs
        self.out1 = OutConv(k, 256, bias)
        self.out2 = OutConv(k, 256, bias)
        self.out3 = OutConv(k, 256, bias)
        self.out4 = OutConv(k, 256, bias)
        self.out5 = OutConv(k, 256, bias)
        self.out6 = OutConv(k, 256, bias)
        
    def forward(self, x):
#         pdb.set_trace()
        x = self.drop(F.relu(self.m_base(x))) 
        
        c1 = self.lat1(self.sfs[0].features) # 128, 28, 28
        c2 = self.sfs[1].features # 256, 14, 14     
        c3 = self.sconv1(x)         # 256, 7, 7
        c4 = self.sconv2(c3)       # 256, 4, 4
        c5 = self.sconv3(c4)      # 256, 2, 2
        c6 = self.sconv4(c5)      # 256, 1, 1
            
        o1c,o1l = self.out1(c1)
        o2c,o2l = self.out2(c2)
        o3c,o3l = self.out3(c3)
        o4c,o4l = self.out4(c4)
        o5c,o5l = self.out5(c5)
#        o6c,o6l = self.out6(p6)
        
        return [torch.cat([o1c,o2c,o3c,o4c,o5c], dim=1),
                torch.cat([o1l,o2l,o3l,o4l,o5l], dim=1)]
    
class MakeModel():
    def __init__(self,model,name='makemodel'):
        self.model,self.name = model,name

    def get_layer_groups(self, precompute):
        lgs = list(split_by_idxs(children(self.model.m_base), [lr_cut]))
        return lgs + [children(self.model)[1:]]
    
def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()]

class BCE_Loss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred, targ):
        t = one_hot_embedding(targ, self.num_classes+1)
        t = V(t[:,:-1].contiguous())#.cpu()
        x = pred[:,:-1]
        w = self.get_weight(x,t)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)/self.num_classes
    
    def get_weight(self,x,t): return None
class FocalLoss(BCE_Loss):
    def get_weight(self,x,t):
        alpha,gamma = 0.25,2.
        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)
        w = alpha*t + (1-alpha)*(1-t)
        return w * (1-pt).pow(gamma)

loss_f = FocalLoss(len(id2cat))
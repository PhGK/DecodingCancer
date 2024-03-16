import torch as tc
import torch.nn as nn
import copy
import torchtuples as tt


class LRP_Linear(nn.Module):
    def __init__(self, inp, outp, gamma=0.01, eps=1e-5):
        super(LRP_Linear, self).__init__()
        self.A_dict = {}
        self.linear = nn.Linear(inp, outp)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        self.gamma = tc.tensor(gamma)
        self.eps = tc.tensor(eps)
        self.rho = None
        self.iteration = None

    def forward(self, x):

        if not self.training:
            self.A_dict[self.iteration] = x.clone()
        return self.linear(x)

    def relprop(self, R):
        device = next(self.parameters()).device

        A = self.A_dict[self.iteration].clone()
        A, self.eps = A.to(device), self.eps.to(device)

        Ap = A.clamp(min=0).detach().data.requires_grad_(True)
        Am = A.clamp(max=0).detach().data.requires_grad_(True)


        zpp = self.newlayer(1).forward(Ap)  
        zmm = self.newlayer(-1, no_bias=True).forward(Am) 

        zmp = self.newlayer(1, no_bias=True).forward(Am) 
        zpm = self.newlayer(-1).forward(Ap) 

        with tc.no_grad():
            Y = self.forward(A).data

        sp = ((Y > 0).float() * R / (zpp + zmm + self.eps * ((zpp + zmm == 0).float() + tc.sign(zpp + zmm)))).data
        sm = ((Y < 0).float() * R / (zmp + zpm + self.eps * ((zmp + zpm == 0).float() + tc.sign(zmp + zpm)))).data

        (zpp * sp).sum().backward()
        cpp = Ap.grad
        Ap.grad = None
        Ap.requires_grad_(True)

        (zpm * sm).sum().backward()
        cpm = Ap.grad
        Ap.grad = None
        Ap.requires_grad_(True)

        (zmp * sm).sum().backward()
        cmp = Am.grad
        Am.grad = None
        Am.requires_grad_(True)

        (zmm * sp).sum().backward()
        cmm = Am.grad
        Am.grad = None
        Am.requires_grad_(True)


        R_1 = (Ap * cpp).data
        R_2 = (Ap * cpm).data
        R_3 = (Am * cmp).data
        R_4 = (Am * cmm).data


        return R_1 + R_2 + R_3 + R_4

    def newlayer(self, sign, no_bias=False):

        if sign == 1:
            rho = lambda p: p + self.gamma * p.clamp(min=0) # Replace 1e-9 by zero
        else:
            rho = lambda p: p + self.gamma * p.clamp(max=0) # same here

        layer_new = copy.deepcopy(self.linear)

        try:
            layer_new.weight = nn.Parameter(rho(self.linear.weight))
        except AttributeError:
            pass

        try:
            layer_new.bias = nn.Parameter(self.linear.bias * 0 if no_bias else rho(self.linear.bias))
        except AttributeError:
            pass

        return layer_new


class LRP_ReLU(nn.Module):
    def __init__(self):
        super(LRP_ReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

    def relprop(self, R):
        return R


class LRP_DropOut(nn.Module):
    def __init__(self, p):
        super(LRP_DropOut, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)

    def relprop(self, R):
        return R

class LRP_cat(nn.Module):
    def __init__(self):
        super(LRP_cat, self).__init__()

    def forward(self, list_of_tensors):
        self.sizes = [tensor.shape[1] for tensor in list_of_tensors]
        return tc.cat(list_of_tensors, axis=1)

    def relprop(self, R):
        splitted_R = tc.split(R,self.sizes,dim=1)
        return splitted_R


class Simple_Model(nn.Module):
    classname = 'Simple Model'
    def __init__(self, data_coll, setting):
        super(Simple_Model, self).__init__()

        self.inp, self.hidden, outp = data_coll.f_nfeatures, int(setting['factor_hidden_nodes']*data_coll.f_nfeatures), 1
        self.hidden_depth = setting['hidden_depth_simple']
        self.layers = nn.Sequential(LRP_DropOut(p = setting['input_dropout']), LRP_Linear(self.inp, self.hidden, gamma=0.01), LRP_ReLU())
        for i in range(self.hidden_depth):
            self.layers.add_module('dropout', LRP_DropOut(p = setting['dropout']))
            self.layers.add_module('LRP_Linear' + str(i + 1), LRP_Linear(self.hidden, self.hidden, gamma=0.01))
            self.layers.add_module('LRP_ReLU' + str(i + 1), LRP_ReLU())
        self.layers.add_module('dropout', LRP_DropOut(p = setting['dropout']))
        self.layers.add_module('LRP_Linear_last', LRP_Linear(self.hidden, outp, gamma=0.01))


        count_parameters(self)
    def forward(self, x):
        return self.layers.forward(x)

    def relprop(self, R):
        assert not self.training, 'relprop does not work during training time'
        for module in self.layers[::-1]:
            R = module.relprop(R)
        return R




def count_parameters(model):
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('{} contains {} trainable parameters'.format(model.classname,nparams))



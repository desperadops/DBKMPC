from ntpath import join
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from collections import OrderedDict
from copy import copy
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
from scipy.integrate import odeint
# physics engine
import pybullet as pb
import pybullet_data
from scipy.io import loadmat, savemat

from ur5e_env import UR5EEnv

#data collect
def Obs(o):
    return np.concatenate((o[:3],o[7:]),axis=0)

class data_collecter():
    def __init__(self,env_name) -> None:
        self.env_name = env_name
        self.env = UR5EEnv(render = False)
        self.Nstates = 15
        # self.uval = 0.12
        self.uval = 0.15
        self.udim = 6
        self.reset_joint_state = np.array(self.env.reset_joint_state)

    def collect_koopman_data(self,traj_num,steps):
        train_data = np.empty((traj_num,steps+1,self.Nstates+self.udim))  # [20000,11,21]
        for traj_i in range(traj_num):  # 每条轨迹
            # noise = (np.random.rand(6)-0.5)*2*0.2  # 噪声
            noise = (np.random.rand(6) - 0.5) * 2 * 0.3  # 噪声
            joint_init = self.reset_joint_state+noise  # 重置关节状态
            joint_init = np.clip(joint_init,self.env.joint_low,self.env.joint_high)
            s0 = self.env.reset_state(joint_init)
            s0 = Obs(s0)
            u10 = (np.random.rand(6)-0.5)*2*self.uval
            train_data[traj_i,0,:]=np.concatenate([s0.reshape(-1),u10.reshape(-1)],axis=0).reshape(-1)
            for i in range(1,steps+1):
                s0 = self.env.step(u10)
                s0 = Obs(s0)
                u10 = (np.random.rand(6)-0.5)*2*self.uval
                train_data[traj_i,i,:]=np.concatenate([s0.reshape(-1),u10.reshape(-1)],axis=0).reshape(-1)
        return train_data # [20000，11，21]
        
#define network
def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega
    
class Network(nn.Module):
    def __init__(self,encode_layers,Nkoopman,u_dim):
        super(Network,self).__init__()
        Layers = OrderedDict()
        for layer_i in range(len(encode_layers)-1):
            Layers["linear_{}".format(layer_i)] = nn.Linear(encode_layers[layer_i],encode_layers[layer_i+1])
            if layer_i != len(encode_layers)-2:
                Layers["relu_{}".format(layer_i)] = nn.ReLU()
        self.encode_net = nn.Sequential(Layers)
        self.Nkoopman = Nkoopman
        self.u_dim = u_dim
        self.lA = nn.Linear(Nkoopman,Nkoopman,bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(u_dim,Nkoopman,bias=False)
        self.lH = nn.Linear(u_dim*Nkoopman,Nkoopman,bias=False)

    def encode(self,x):
        return torch.cat([x,self.encode_net(x)],axis=-1)
    
    def forward(self,x,u,batchsize):
        return self.lA(x)+self.lB(u)+self.lH(self.cal_bilinear(x, u, batchsize))
        # return self.lA(x) + self.lB(u)
    def cal_bilinear(self,x,u,batchsize):
        xu1 = x*torch.reshape(u[:, 0], (batchsize, 1))
        xu2 = x * torch.reshape(u[:, 1], (batchsize, 1))
        xu3 = x * torch.reshape(u[:, 2], (batchsize, 1))
        xu4 = x * torch.reshape(u[:, 3], (batchsize, 1))
        xu5 = x * torch.reshape(u[:, 4], (batchsize, 1))
        xu6 = x * torch.reshape(u[:, 5], (batchsize, 1))
        # xu7 = x * torch.reshape(u[:, 6], (batchsize, 1))

        return torch.cat([xu1, xu2, xu3, xu4, xu5, xu6], axis=-1)

#loss function
def Klinear_loss(data,net,mse_loss,u_dim=1,gamma=0.99,Nstate=4,all_loss=0,batchsize=512):
    train_traj_num,steps,NKoopman = data.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.DoubleTensor(data).to(device)  # 变成double类型
    X_current = net.encode(data[:,0,0:Nstate])
    beta = 1.0
    beta_sum = 0.0
    loss = torch.zeros(1,dtype=torch.float64).to(device)
    for i in range(steps-1):  # 10 步计算 10次
        X_current = net.forward(X_current,data[:,i,Nstate:],batchsize)
        beta_sum += beta
        if not all_loss:
            loss += beta*mse_loss(X_current[:,:Nstate],data[:,i+1,Nstate:])
        else:
            Y = net.encode(data[:,i+1,0:Nstate])
            loss += beta*mse_loss(X_current,Y)
        beta *= gamma
    loss = loss/beta_sum
    return loss

#
# def Klinear_loss_1(data , net, mse_loss, u_dim=1, gamma=0.99, Nstate=4, all_loss=0, batchsize=512):
#     train_traj_num,steps,NKoopman = data.shape
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     data = torch.DoubleTensor(data).to(device)  # 变成double类型
#     X_current = net.encode(data[:,0,0:Nstate])
#     beta = 1.0
#     beta_sum = 0.0
#     loss = torch.zeros(1,dtype=torch.float64).to(device)
#     for i in range(steps-1):  # 10 步计算 10次
#         X_current = net.forward(X_current,data[:,i,Nstate:],batchsize)
#         beta_sum += beta
#         if not all_loss:
#             loss += beta*mse_loss(X_current[:,:Nstate],data[:,i+1,Nstate:])
#         else:
#             Y = net.encode(data[:,i+1,0:Nstate])
#             loss += beta*mse_loss(X_current,Y)
#         beta *= gamma
#     loss = loss/beta_sum
#     return loss


def Stable_loss(net,Nstate):
    x_ref = np.zeros(Nstate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_ref_lift = net.encode_only(torch.DoubleTensor(x_ref).to(device))
    loss = torch.norm(x_ref_lift)
    return loss

def Eig_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = net.lA.weight
    c = torch.linalg.eigvals(A).abs()-torch.ones(1,dtype=torch.float64).to(device)
    mask = c>0
    loss = c[mask].sum()
    return loss

def H_loss(net):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # B = net.lB.weight
    H = net.lH.weight
    target = torch.zeros([net.Nkoopman, net.u_dim*net.Nkoopman], dtype=torch.float64).to(device)
    # target1 = torch.zeros([net.Nkoopman, net.u_dim], dtype=torch.float64).to(device)
    l1loss = torch.nn.L1Loss()
    loss = l1loss(H, target).to(device)
           # + l1loss(B, target1).to(device)
    return loss

def train(env_name,train_steps = 300000,suffix="",all_loss=0,\
            encode_dim = 20,layer_depth=3,e_loss=1,gamma=0.5):
    np.random.seed(98)
    # Ktrain_samples = 1000
    # Ktest_samples = 1000
    Ktrain_samples = 50000
    Ktest_samples = 20000
    Ksteps = 10
    Kbatch_size = 512
    u_dim = 6
    #data prepare
    data_collect = data_collecter(env_name)  # 数据收集
    Ktest_data = data_collect.collect_koopman_data(Ktest_samples,Ksteps)  # [20000,11,21]
    print("test data ok!")
    Ktrain_data = data_collect.collect_koopman_data(Ktrain_samples,Ksteps)  # [50000,11,21]
    print("train data ok!")

    # raise NotImplementedError
    in_dim = Ktest_data.shape[-1]-u_dim  # 15
    Nstate = in_dim  # 15
    layer_width = 128
    layers = [in_dim]+[layer_width]*layer_depth+[encode_dim]  # [15 ,128, 128, 128, 20]
    Nkoopman = in_dim+encode_dim  # 35
    print("layers:",layers)
    net = Network(layers,Nkoopman,u_dim)
    # print(net.named_modules())
    eval_step = 1000
    learning_rate = 1e-3  # 学习率0.001

    beta = 0.01

    if torch.cuda.is_available():
        net.cuda() 
    net.double()  #
    mse_loss = nn.MSELoss()  # mse误差
    optimizer = torch.optim.Adam(net.parameters(),
                                    lr=learning_rate)
    for name, param in net.named_parameters():
        print("model:",name,param.requires_grad)
    #train
    eval_step = 1000
    best_loss = 1000.0
    best_state_dict = {}
    subsuffix = suffix+"KK_"+env_name+"layer{}_edim{}_eloss{}_gamma{}_aloss{}_hloss{}".format(layer_depth,encode_dim,e_loss,gamma,all_loss,beta)
    logdir = "Data_test/"+suffix+"/"+subsuffix
    if not os.path.exists( "Data_test/"+suffix):
        os.makedirs("Data_test/"+suffix)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    writer = SummaryWriter(log_dir=logdir)  # 写入tensorboard
    for i in range(train_steps):  # 跑200000个批次（epoch）
        #K loss
        Kindex = list(range(Ktrain_samples)) # 0 1 2.... 49999
        random.shuffle(Kindex)  # 将列表中的顺序打乱
        X = Ktrain_data[Kindex[:Kbatch_size],:,:]  # 取一个batch的数据
        Kloss = Klinear_loss(X,net,mse_loss,u_dim,gamma,Nstate,all_loss,Kbatch_size)
        Eloss = Eig_loss(net)
        Hloss = H_loss(net)
        loss = Kloss+Eloss+beta*Hloss if e_loss else Kloss+beta*Hloss  # 如果e_loss为真，loss = Kloss+Eloss 如果e_loss为假，loss = Kloss
        # loss = Kloss + Eloss if e_loss else Kloss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        writer.add_scalar('Train/Kloss',Kloss,i)
        writer.add_scalar('Train/Eloss',Eloss,i)
        writer.add_scalar('Train/Hloss',Hloss,i)
        writer.add_scalar('Train/loss',loss,i)
        # print("Step:{} Loss:{}".format(i,loss.detach().cpu().numpy()))
        if (i+1) % eval_step == 0:
            #K loss
            Kloss = Klinear_loss(Ktest_data, net, mse_loss, u_dim, gamma, Nstate, all_loss, batchsize=Ktest_samples)
            # Kloss_1 = Klinear_loss_1(Ktest_data, net, mse_loss, u_dim, 1.0, Nstate, all_loss, batchsize=Ktest_samples)
            Eloss = Eig_loss(net)
            Hloss = H_loss(net)
            loss = Kloss+Eloss+beta*Hloss if e_loss else Kloss+beta*Hloss
            # loss = Kloss + Eloss if e_loss else Kloss
            Kloss = Kloss.detach().cpu().numpy()
            # Kloss_1 = Kloss_1.detach().cpu().numpy()
            Eloss = Eloss.detach().cpu().numpy()
            Hloss = Hloss.detach().cpu().numpy()
            loss = loss.detach().cpu().numpy()
            writer.add_scalar('Eval/Kloss',Kloss,i)
            writer.add_scalar('Eval/Eloss',Eloss,i)
            writer.add_scalar('Eval/Hloss', Hloss, i)
            writer.add_scalar('Eval/loss',loss,i)
            # if loss < best_loss:
            if Kloss < best_loss:
                best_loss = copy(Kloss)
                best_state_dict = copy(net.state_dict())
                Saved_dict = {'model':best_state_dict,'layer':layers}
                torch.save(Saved_dict,"Data_test/"+subsuffix+".pth")
                # torch.save(Saved_dict, "" + subsuffix + ".pth")
            print("Step:{} Eval-loss{} K-loss:{} E-loss:{} H-loss:{}".format(i,loss,Kloss,Eloss,Hloss))
            # print("K-loss_1:{} ".format(Kloss_1))
            # print("-------------END-------------")
    print("END-best_loss{}".format(best_loss))
    

def main():
    train(args.env,suffix=args.suffix,all_loss=args.all_loss,\
        encode_dim=args.encode_dim,layer_depth=args.layer_depth,\
            e_loss=args.eloss,gamma=args.gamma)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",type=str,default="UR5E")
    parser.add_argument("--suffix",type=str,default="")
    parser.add_argument("--all_loss",type=int,default=1)
    parser.add_argument("--eloss",type=int,default=0)
    parser.add_argument("--gamma",type=float,default=0.8)
    parser.add_argument("--encode_dim",type=int,default=20)
    parser.add_argument("--layer_depth",type=int,default=3)
    args = parser.parse_args()
    main()


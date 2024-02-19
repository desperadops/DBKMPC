import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from cvxopt import solvers, matrix
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# import gym
# import matplotlib.pyplot as plt
import random
import argparse
from collections import OrderedDict
from copy import copy
import Learn_Koopman_with_BIlinear_UR5E as lka
import scipy
import scipy.linalg
from scipy.integrate import odeint
import pybullet as pb
import pybullet_data
import math
import sys
import datetime
import csv

sys.path.append("../utility")

from ur5e_env import UR5EEnv

# a = open("test.csv", 'w', newline='')
# data_writer1 = csv.writer(a)
# for row in train_data:
#     data_writer1.writerow(row)


env_name = "Franka"
layer_depth = 3
encode_dim = 20
gamma = 0.8
all_loss = 1
eloss = 0
env = UR5EEnv(render=True)

in_dim = env.Nstates  # 15
u_dim = env.udim  # 6
# dict = torch.load('epoch_4999_train_model.pth')
dicts = torch.load('Data_test/KK_Frankalayer3_edim20_eloss0_gamma0.8_aloss1_hloss0.01.pth')
state_dict = dicts["model"]
Elayer = dicts["layer"]
# print(layers)
Nkoopman = encode_dim + in_dim  # 35
net = lka.Network(Elayer, Nkoopman, u_dim)
net.load_state_dict(state_dict)
device = torch.device("cpu")
net.to(device)
net.double()

Samples = 20000  # Number of random initial conditions for both training and testing data
ts = 0.02  # time spacing between training state measurements
tFinal = 0.3  # time horizon --- used in measuring error

NKoopman = encode_dim + in_dim  # Number of basis functions (keep inside [2,4]) 35
Nstates = in_dim  # Number of system states  15
Ncontrol = u_dim  # Number of system inputs  6

timeSteps = round(tFinal / ts) + 1


def Psi_k(s, u, net):  # Evaluates basis functions Ψ(s(t_k))
    psi = np.zeros([NKoopman + Ncontrol, 1])
    ds = net.encode(torch.DoubleTensor(s)).detach().cpu().numpy()
    psi[:NKoopman, 0] = ds
    psi[NKoopman:] = u
    return psi


def Psi_o(s, net):  # Evaluates basis functions Ψ(s(t_k))
    psi = np.zeros([NKoopman, 1])
    ds = net.encode(torch.DoubleTensor(s)).detach().cpu().numpy()
    psi[:NKoopman, 0] = ds
    return psi


def A_and_G(s_1, s_2, u, net):  # Uses measurements s(t_k) & s(t_{k+1}) to calculate A and G
    A = np.dot(Psi_k(s_2, u, net), Psi_k(s_1, u, net).T)
    G = np.dot(Psi_k(s_1, u, net), Psi_k(s_1, u, net).T)
    return A, G


def A_and_G_o(s_1, s_2, u, net):  # Uses measurements s(t_k) & s(t_{k+1}) to calculate A and G
    A = np.dot(Psi_o(s_2, u, net), Psi_o(s_1, u, net).T)
    G = np.dot(Psi_o(s_1, u, net), Psi_o(s_1, u, net).T)
    return A, G


def Obs(o):
    noise = np.random.randn(3) * 1e-1
    return np.concatenate((o[:3] + noise, o[7:]), axis=0), noise


Ad = state_dict['lA.weight'].cpu().numpy()
Bd = state_dict['lB.weight'].cpu().numpy()
Hd = state_dict['lH.weight'].cpu().numpy()
eig = scipy.linalg.eigvals(Ad)
# print("Eigs of the matrix:{}".format(eig))
print("The max eigen of Kd is {}".format(max(eig)))
print("The max eigen of Kd is {}".format(eig))
env.reset()
nStates = 9
accuracy_invKin = 0.000001
T = 6 * 10  # time horizon
t = 1.6 + 0.02 * np.linspace(0, T * 5, T * 50 + 1)  # time steps
Steps = len(t) - 1

a = 0.4  # scaling parameter of Fig. 8 area
x = 0.4 * np.ones((len(t), 1))
z = np.expand_dims(0.44 + 2 * a * np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2), axis=1)
y = np.expand_dims(a * np.cos(t) / (1 + np.sin(t) ** 2), axis=1)


# plt.plot(y, z)
# plt.axis('equal')
# plt.xlabel('y (m)')
# plt.ylabel('z (m)')


#
def accurateCalculateInverseKinematics(kukaId, endEffectorId, targetPos, threshold, maxIter):
    """
    Calculates the joint poses given the End Effector location using inverse kinematics
    Note: It changes the Franka configuration during the optimization to the desired configuration

    Input:
    kukaId : Object that represents the Franka system
    endEffectorId :
    targetPos :
    threshold : accuracy threshold
    maxIter : maximum iterations to fine tune solution

    Output:
    jointPoses: The angles of the 7 joints of the Franka
    """
    numJoints = 6
    closeEnough = False
    iter = 0
    dist2 = 1e30
    # init_pose = pb.getQuaternionFromEuler([0, math.pi/2, 0])
    while (not closeEnough and iter < maxIter):
        jointPoses = pb.calculateInverseKinematics(kukaId, endEffectorId, targetPos, )

        for i in range(numJoints):
            pb.resetJointState(kukaId, i + 1, jointPoses[i])
        ls = pb.getLinkState(kukaId, endEffectorId)
        newPos = ls[4]
        diff = [targetPos[0] - newPos[0], targetPos[1] - newPos[1], targetPos[2] - newPos[2]]
        dist2 = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2])
        closeEnough = (dist2 < threshold)
        iter = iter + 1
    return jointPoses[:6]



JointAngles_Fig8 = np.empty((len(t), 6))
JointAngles_Fig8[:] = np.NaN

for i in range(len(t)):
    JointAngles_Fig8[i, :] = accurateCalculateInverseKinematics(env.robot, env.ee_id, [x[i], y[i], z[i]],
                                                                accuracy_invKin, 10000)
# states_des = np.concatenate((x, y, z, JointAngles_Fig8, np.zeros((len(y), 7))), axis=1)
states_des = np.concatenate((x, y, z, JointAngles_Fig8, 0.001 * np.ones((len(y), 6))), axis=1)

np.savetxt("states_des_moveit.txt", states_des)
# states_des = np.concatenate((x,y,z), axis = 1)


def cal_matrices(A, B, Q, R, F, N):
    n = A.shape[0]
    p = B.shape[1]

    M = np.vstack((np.eye((n)), np.zeros((N * n, n))))
    C = np.zeros(((N + 1) * n, N * p))
    tmp = np.eye(n)

    for i in range(N):
        rows = i * n + n
        C[rows:rows + n, :] = np.hstack((np.dot(tmp, B), C[rows - n:rows, 0:(N - 1) * p]))
        tmp = np.dot(A, tmp)
        M[rows:rows + n, :] = tmp

    Q_bar_be = np.kron(np.eye(N), Q)
    Q_bar = scipy.linalg.block_diag(Q_bar_be, F)
    R_bar = np.kron(np.eye(N), R)

    G = np.matmul(np.matmul(M.transpose(), Q_bar), M)
    E = np.matmul(np.matmul(C.transpose(), Q_bar), M)
    H = np.matmul(np.matmul(C.transpose(), Q_bar), C) + R_bar

    return H, E


def Prediction(M, T):
    sol = solvers.qp(M, T, kktsolver='ldl', options={'kktreg': 1e-15})
    U_thk = np.array(sol["x"])
    u_k = U_thk[0:u_dim, :]
    return u_k


trail = np.zeros([3001, Nstates])  # 3001 15
Trail = np.zeros([3001, NKoopman])  # 3001 35
trail = states_des
trail = torch.tensor(trail, dtype=torch.float64, requires_grad=False)
Trail = net.encode(trail).detach().numpy()
initial_data = Trail[0][:]
control_trail = np.zeros([3000, Nstates])
# deired_tra = np.zeros([100, NKoopman])

A = Ad
h = np.array_split(Hd, NKoopman, axis=1)
H = np.zeros([NKoopman, u_dim])
for i in range(NKoopman):
    H = h[i] * initial_data[i] + H
B = Bd + H
Q = np.eye(NKoopman) * 1000000
Q[9:] = 0
Q[3:9] = Q[3:9]
Q[0:3] = Q[0:3]
F = np.eye(NKoopman) * 1000000
F[9:] = 0
F[3:9] = F[3:9]
F[0:3] = F[0:3]
R = np.eye(u_dim) * 0.1

N = 5

G = np.eye(N*u_dim)
hhh = np.ones([N*u_dim]) * 0.3
G = matrix(G)
hhh = matrix(hhh)
K_steps = 3001  # 预测步数
X_k = np.zeros((NKoopman, K_steps))
X_k[:, 0] = Trail[0][:]
# X_k[:, 0] = initial_data
U_k = np.zeros((u_dim, K_steps))


# state_min = np.array()
state_max = np.zeros((35, 1))
state_max[2] = 0.7
state_max[1] = 0.26
state_max[0] = 0.95

state = env.reset()
for i, jnt in enumerate(states_des[0, 3:8]):
    pb.resetJointState(env.robot, i + 1, jnt)
JointAngles_Fig8 = accurateCalculateInverseKinematics(env.robot, env.ee_id, [x[0, 0], y[0, 0], z[0, 0]],
                                                      accuracy_invKin, 10000)
for i, jnt in enumerate(JointAngles_Fig8[0:5]):
    pb.resetJointState(env.robot, i + 1, jnt)

for k in range(1, K_steps):
    # starttime = datetime.datetime.now()
    x_kshort = X_k[:, k - 1].reshape(NKoopman, 1)
    u_kshort = U_k[:, k - 1].reshape(6, 1)

    M, C = cal_matrices(A, B, Q, R, F, N)
    M = matrix(M)

    T = np.dot(C, x_kshort)
    T = matrix(T)
    for i in range(u_dim):
        U_k[i, k - 1] = Prediction(M, T)[i, 0]
    # b = U_k[:, k-1]
    X_knew = env.step(U_k[:, k - 1])
    control_trail[k - 1] = X_knew
    # noise = np.random.randn(3) * 1e-2
    # X_knew = np.concatenate((X_knew[:3] + noise, X_knew[3:]), axis=0)
    # time.sleep(1.0 / 600000.)
    # time.sleep(0.5)
    X_knew = torch.tensor(X_knew, dtype=torch.float64, requires_grad=False)
    X_knew = net.encode(X_knew).detach().numpy()

    # m = Trail[k]
    X_next = X_knew - Trail[k]
    for j in range(NKoopman):
        X_k[j, k] = X_next[j]
    H = 0
    for i in range(NKoopman):
        H = h[i] * X_knew[i] + H
    B = Bd + H

    # z_max = state_max[2] - np.matmul(A, X_knew.reshape(35, 1))[2]
    # y_max = state_max[1] - np.matmul(A, X_knew.reshape(35, 1))[1]
    yz_max = state_max[0] - np.matmul(A, X_knew.reshape(35, 1))[2] - np.matmul(A, X_knew.reshape(35, 1))[1]

    # hhh[2, 0] = z_max
    # hhh[1, 0] = y_max
    hhh[0, 0] = yz_max

    G[0, :6] = B[1] + B[2]
    # G[2, :6] = B[2]
    # G[1, :6] = B[1]

    # endtime = datetime.datetime.now()
    # print(endtime-starttime)

b = X_k[0:9, 1:]
c = b ** 2
a = np.sum(c, axis=0)
a = np.sum(np.sqrt(a))
# np.savetxt("states_des.txt", states_des)
np.savetxt("control_trail.txt", control_trail)
np.savetxt("X_k.txt", X_k)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = 'FangSong'
plt.plot(states_des[:, 1], states_des[:, 2], label='期望轨迹')
plt.plot(control_trail[:, 1], control_trail[:, 2], label='本方法控制的实际轨迹')
plt.legend()
plt.show()
JointAngles_Fig8 = accurateCalculateInverseKinematics(env.robot, env.ee_id, [-0.5, 0, 0.3],
                                                      accuracy_invKin, 10000)
print(JointAngles_Fig8)

#goal is to compare the output of an ofc model

import numpy as np
import matplotlib.pyplot as plt
N = 30
A = np.array([[1,0,1,0],[0,1,0,1],[0,0,0.5,0],[0,0,0,0.5]])
B = np.array([[0,0],[0,0],[1,0],[0,1]])
QN = np.zeros((4,4))
QN[0,0]=5
QN[1,1]=5
QN[2,2]=1
QN[3,3]=1

R = np.array([[10**-4,0],[0,10**-4]])
L = np.zeros((N,2,4))
S = np.zeros((N,4,4))
Q = np.zeros((N,4,4))

Q[N - 1, :, :] = QN
S[N - 1, :, :] = QN

for i in range(N - 1, 0, -1):
  L[i,:,:]=np.linalg.solve(R+B.T@S[i,:,:]@B,B.T@S[i,:,:]@A)
  S[i-1,:,:]=A.T@S[i,:,:]@(A-B@L[i,:,:])
X = np.zeros((N, 4, 1))


X[0, :, :] = [[0.5], [1], [0], [0]]

Xi = np.random.normal(loc=0, scale=10 ** -4, size=(N, 4, 1))
for j in range(0, N - 1):
  X[j+1,:,:]=(A-B@L[j,:,:])@X[j,:,:]+Xi[j,:,:]


fig, ax=plt.subplots()
ax.plot(X[:,0,:],X[:,1,:],'r')
fig, ax=plt.subplots()
ax.plot(range(N),X[:,0,:],'b')
ax.plot(range(N),X[:,1,:],'r')
fig, ax=plt.subplots()
ax.plot(range(N),X[:,2,:],'r')
ax.plot(range(N),X[:,3,:],'b')
plt.show()
actions = np.zeros((N,2))
for ii in range(N):
    actions[ii,:] = L[ii,:,:]@X[ii,:,:]


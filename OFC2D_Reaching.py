import numpy as np
import matplotlib.pyplot as plt
N = 50  # Nb of steps

dt = 0.01
kv = 0.1
tau = 0.05
lamb = 0
ns = 6

###############################
## Complete the code below ####
###############################


A = np.array([[1,0,dt,0,0,0],[0,1,0,dt,0,0],[0,0,1-kv*dt,0,dt,0],[0,0,0,1-kv*dt,0,dt],[0,0,0,0,1-dt/tau,0],[0,0,0,0,0,1-dt/tau]])
B = np.zeros((6,2))
B[4,:]=np.array([dt/tau,0])
B[5,:]=np.array([0,dt/tau])
w1 = 10
w2 = 10
w3 = 0.1
w4 = 0.1
QN = np.zeros((6,6))
QN[0,0]=w1
QN[1,1]=w2
QN[2,2]=w3
QN[3,3]=w4
# We set the R matrix as follows, later on you can change it to see its effect on the controller
R = np.array([(10 ** -4, 0), (0, 10 ** -4)])
L = np.zeros((N, 2, ns))
S = np.zeros((N, ns, ns))
Q = np.zeros((N, ns, ns))

###############################
## Complete the code below ####
###############################
# (hint : fill in L and S matrices in the backward loop)
Q[N - 1, :, :] = QN
S[N - 1, :, :] = QN

for i in range(N - 1, 0, -1):
  L[i,:,:]=np.linalg.solve(R+B.T@S[i,:,:]@B,B.T@S[i,:,:]@A)
  S[i-1,:,:]=A.T@S[i,:,:]@(A-B@L[i,:,:])
X = np.zeros((N, ns, 1))

#Change the first entries of the vector below to investigate different starting position
print(L[45,:,:])
X[0, :, :] = [[0.2], [0.3], [0], [0], [0], [0]]

#Computation of the motor noise
Xi = np.random.normal(loc=0, scale=10 ** -4, size=(N, 6, 1))

###############################
## Complete the code below ####
###############################
for j in range(0, N - 1):
  X[j+1,:,:]=(A-B@L[j,:,:])@X[j,:,:]+Xi[j,:,:]
###############################
## Complete the code below ####
###############################

#Create a representation of positions and speeds with respect to time and characterise their evolution
fig, ax=plt.subplots()
ax.plot(X[:,0,:],X[:,1,:],'r')
fig, ax=plt.subplots()
ax.plot(range(N),X[:,0,:],'b')
ax.plot(range(N),X[:,1,:],'r')
fig, ax=plt.subplots()
ax.plot(range(N),X[:,2,:],'r')
ax.plot(range(N),X[:,3,:],'b')

#Initialize the state estimation... What is the size of hte matrix? How would you complete the information corresponding to the first time step?

Xhat = np.zeros_like(X)
Xhat[0, :, :] = X[0,:,:] + np.random.normal(loc=0, scale=10 ** -6, size=(6, 1))

#Initialization of the command and observable
Y = np.zeros((N, ns, 1))
U = np.zeros((N,2,1))


#Initialization of the covariance matrix of the state, how would you initialize the first covariance matrix?
Sigma = np.zeros((N, ns, ns))
Sigma[0,:,:] = np.random.normal(loc=0, scale=10 ** -2, size=(1, ns, 1))


#Some more initialization (nothing to do for you here)
K = np.zeros((N, ns, ns))
H = np.eye(ns)
Xi = np.random.normal(loc=0, scale=10 ** -4, size=(N, ns, 1))
Omega = np.random.normal(loc=0, scale=10 ** -2, size=(N, ns, 1))
oXi = 0.1 * (B @ B.T)
oOmega = 0.1 * np.max(np.max(oXi)) * np.eye(ns)

#Fill in the following loop to complete
#
# state evolution
# observatoin evolutino
# computation of K and Sigma
# computation of the command
# evolution of the state estimation

for j in range(0, N - 1):
  X[j+1,:,:]=A@X[j,:,:]-B@L[j,:,:]@Xhat[j,:,:]+Xi[j,:,:]
  Y[j+1,:,:] = H@X[j,:,:]+Omega[j+1,:,:]

  K[j,:,:] = A@Sigma[j,:,:]@H.T@np.linalg.inv(H@Sigma[j,:,:]@H.T+oOmega)
  Sigma[j+1,:,:] = oXi + (A-K[j,:,:]@H)@Sigma[j,:,:]@A.T

  Xhat[j+1,:,:] = (A-B@L[j,:,:])@Xhat[j,:,:] + K[j,:,:]@(Y[j,:,:]-H@Xhat[j,:,:])

#Plot the time evolution of the state, its observation and its estimation.. What do you observe?
fig, ax=plt.subplots()
ax.plot(X[:,0,:],X[:,1,:],'r')
fig, ax=plt.subplots()
ax.plot(range(N),X[:,0,:],'b')
ax.plot(range(N),X[:,1,:],'r')
ax.plot(range(N),Xhat[:,0,:],'b:')
ax.plot(range(N),Xhat[:,1,:],'r:')



fig, ax=plt.subplots()
ax.plot(range(N),X[:,2,:],'r')
ax.plot(range(N),X[:,3,:],'b')
plt.show()
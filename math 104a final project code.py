import numpy as np
import time
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
import math

def u(x,y):
    res = math.sin(x*y*(1-x)*(1-y))
    #res=x*x+y*y
    return res
def f(x,y):
    res = -2*x*(1-x)*math.cos(x*y*(1-x)*(1-y)) -((1- 2*y) ** 2)*((x*(1-x))**2) * math.sin(x*y*(1-x)*(1-y))
    res = res - 2*y*(1- y)*math.cos(x*y*(1-x)*(1-y)) - ((1- 2*x) ** 2)*((y*(1-y))**2) * math.sin(x*y*(1-x)*(1-y))
    #res=-4
    return res

def Gauss(A,b):
    n = b.shape[0]
    y = np.zeros(b.shape[0])
    for i in range(n-1):
        for j in range(i+1,n):
            if A[j,i]==0:
                continue
            A[j,i] = A[j,i] / A[i,i] 

            # for m in range(i+1,n):
            #     if A[i,m] !=0:
            #         A[j,m] =  A[j,m] - A[j,i] * A[i,m]  
            b[j] = b[j] - A[j,i] * b[i]  
            A[j] = A[j] - A[j,i] * A[i]
            
    for i in range(n-1,-1,-1):
        y[i] = (b[i] - sum(A[i,i+1:n]*y[i+1:n])) / A[i,i]
    return y.reshape(-1,1)

def index(n):
    h=1.0/n
    b=np.zeros((n+1,n+1))
    x_real=np.zeros((n+1,n+1))
    for i in range(n+1):
        for j in range(n+1):
            x_real[i,j]=u(i*h,j*h)
            b[i,j]=f(i*h,j*h)
    return b,x_real

def buildA(n):
    h= 1.0/n
    A = np.zeros(((n-1)*(n-1), (n-1)*(n-1)))
    A[0,0]= 0
    A[0,1] = 1 / (h*h)
    A[0,n-1] = -1 / (h*h)
    for i in range(n-1):
        for j in range(n-1):
            A[i*(n-1)+j,i*(n-1)+j] = -4 / (h*h)
            if i == 0:
                if j != 0:
                    A[i*(n-1)+j,i*(n-1)+j-1]= 1 / (h*h)
                if j != n-2:
                    A[i*(n-1)+j,i*(n-1)+j+1] = 1 / (h*h)
                A[i*(n-1)+j, (i+1)*(n-1)+j] = 1 / (h*h)
            elif i == n-2:
                if j != 0:
                    A[i*(n-1)+j,i*(n-1)+j-1]= 1 / (h*h)
                if j != n-2:
                    A[i*(n-1)+j,i*(n-1)+j+1] = 1 / (h*h)
                A[i*(n-1)+j, (i-1)*(n-1)+j] = 1 / (h*h)
            elif j==0:
                A[i*(n-1)+j,i*(n-1)+j+1] = 1 / (h*h)
                # try:
                A[i*(n-1)+j, (i+1)*(n-1)+j] = 1 / (h*h)
                # except:
                #     print(i,j)
                A[i*(n-1)+j, (i-1)*(n-1)+j] = 1 / (h*h)
            elif j == n-2:
                A[i*(n-1)+j,i*(n-1)+j-1] = 1 / (h*h)
                A[i*(n-1)+j, (i+1)*(n-1)+j] = 1 / (h*h)
                A[i*(n-1)+j, (i-1)*(n-1)+j] = 1 / (h*h)
            else:
                A[i*(n-1)+j,i*(n-1)+j-1] = 1 / (h*h)
                A[i*(n-1)+j,i*(n-1)+j+1] = 1 / (h*h)
                A[i*(n-1)+j, (i+1)*(n-1)+j] = 1 / (h*h)
                A[i*(n-1)+j, (i-1)*(n-1)+j] = 1 / (h*h)
    return A
n = 9
for n in [10,20,40,80]:
    b,real = index(n)
    b = b[1:-1,1:-1]
    real = real[1:-1,1:-1]
    real = real.reshape(-1,1)
    b = b.reshape(-1)
    A = buildA(n)
    fake = Gauss(A,b)
    b = b.reshape(-1,1)
    err=np.linalg.norm((real-fake).reshape(-1), ord=np.inf, axis=None, keepdims=False)
    print("when h = j = 1 / {}, the error is {}.".format(n,err))
sub1 = np.zeros(n-1)
sub2 = np.zeros(n-1)
for i in range(n-1):
    sub1[i] = fake[(n-1) * i + 39]
    sub2[i] = real[(n-1) * i + 39]

plt.subplot(2,1,1)
plt.title("Numerical results")
plt.plot(np.linspace(0,1,81)[1:-1],sub1)
plt.subplot(2,1,2)
plt.title("Actual results")
plt.plot(np.linspace(0,1,81)[1:-1],sub2)
plt.show()
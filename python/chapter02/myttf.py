# -*- encoding: utf-8 -*-
'''
@File    :   myttf.py
@Time    :   2023/01/13 17:05:52
@Author  :   jolly 
@Version :   python3.8
@Contact :   jmlw8023@163.com
@Reference_Linking  :
'''

# import packets
import numpy as np
import math as ma
import cmath
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def f(t,N):
    x=ma.cos(2*ma.pi/N*t)+0.5*ma.cos(2*2*ma.pi/N*t)+0.8*ma.cos(5*2*ma.pi/N*t)
    return x


r=int(input('输入5 ~ 10 整数:'))
N=2**r
n=N/2
c=0
nu=0

t=np.arange(0,N,1);y=np.zeros((N,r));p=np.zeros((N,r))
x=np.zeros((N,1));x1=[]
for i in range(N):
    m=list(bin(i))
    for j in range(len(m)-2):
        y[i,r-1-j]=float(m[len(m)-j-1])
for l in range(1,r+1):
    z=np.zeros((N,r))
    for no in range(r):
        if r-no-1-(r-l)>=0:
            z[:,r-no-1]=y[:,r-no-1-(r-l)]
    for nk in range(N):
        c=0
        for mk in range(r):
            c=c+z[nk,mk]*2**mk
        p[nk,l-1]=c
for nk in range(N):
    x[nk,0]=f(nk,N)
x=x+0j
for j in range(N):
    x1.append(j)
y1=np.array([x1]);
for io in range(1,r+1):
    y1=y1.reshape((2**io,-1))
    b=int(y1.shape[0]/2);l=y1.shape[1];lp=0
    mn=np.zeros((2,int(N/2)))
    for nk in range(l):
        for nl in range(b):
            bg=2*(nl+1)-1
            bv=2*(nl+1)-2
            mn[0,lp]=y1[bv,nk]
            mn[1,lp]=y1[bg,nk]
            lp=lp+1
    a1=np.lexsort(mn[::-1,:])
    mn=mn[:,a1]
    a2=np.lexsort(mn[0:2,:])
    mn=mn[:,a2];nk1=np.zeros((N,1));
    for lop in range(int(N)):
         nk1[lop,0]=x[lop,0]
    for nk in range(int(N/2)):
         pg=[]
         for gg in mn[:,nk]:
            pg.append(gg)
         x[int(pg[0]),0]=nk1[int(pg[0]),0]+cmath.exp(-2*ma.pi*1j/int(N)*p[int(pg[0]),io-1])*nk1[int(pg[1]),0]
         x[int(pg[1]),0]=nk1[int(pg[0]),0]+cmath.exp(-2*ma.pi*1j/int(N)*p[int(pg[1]),io-1])*nk1[int(pg[1]),0]
x=abs(x)
bf=x.reshape(1,-1)
x=np.arange(0,N,1)
x=np.array([x])
y=[];m=0
zk=np.zeros((1,int(N)))
for ba in p[:,-1]:
    zk[0,int(ba)]=bf[0,m]
    m=m+1
for bb in zk:
    for bh in bb:
        y.append(bh)
        for nm in range(100):
            y.append(0)
vf=[]
for nn in x:
    for vvc in nn:
        vf.append(vvc)
        for llk in range(100):
            vf.append(vvc+llk*0.01)


plt.plot(vf,y)

plt.xlabel('K值')
plt.ylabel('频谱幅度')
plt.title('FFT频域')
plt.show()


mk, po = [], []
for bg in range(int(N)+1):
    mk.append(f(bg,N))
    po.append(bg)


mk1, pno = [], []
for bg2 in range(0,int(N)*25):
    mk1.append(f(bg2,N))
    pno.append(bg2)


plt.plot(pno,mk1)
plt.scatter(po,mk)
plt.ylabel('x')
plt.xlabel('t')
plt.title('FFT时域')
plt.show()
plt.scatter(po,mk)
plt.plot(po,mk)
plt.ylabel('x')
plt.xlabel('t')
plt.title('FFT时域采样点')
plt.show()





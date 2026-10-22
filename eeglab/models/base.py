import copy
import numpy as np
from scipy.special import expit, logsumexp


class DenseClassifier:
    name = 'base'
    binary = False

    def __init__(self, n_features, n_classes, hidden=(32,16,8), seed=42):
        if n_classes < 2 or (self.binary and n_classes != 2):
            raise ValueError('Model Sigmoid membutuhkan tepat dua kelas; model lain minimal dua kelas.')
        rng=np.random.default_rng(seed)
        sizes=[n_features,*hidden,1 if self.binary else n_classes]
        self.weights=[rng.normal(0,np.sqrt(1/a),(a,b)) for a,b in zip(sizes[:-1],sizes[1:])]
        self.biases=[np.zeros((1,b)) for b in sizes[1:]]
        self.hidden=tuple(hidden);self.n_classes=n_classes;self.n_features=n_features

    def activation(self,z):raise NotImplementedError
    def derivative(self,z):raise NotImplementedError

    def forward(self,x):
        values=[x]; pre=[]
        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            z=values[-1]@w+b;pre.append(z)
            if i<len(self.weights)-1:a=self.activation(z)
            elif self.binary:a=expit(z)
            else:a=np.exp(z-logsumexp(z,axis=1,keepdims=True))
            values.append(a)
        return values,pre

    def loss_gradients(self,x,y):
        values,pre=self.forward(x);n=len(x)
        if self.binary:
            target=y[:,None]
            loss=np.mean(np.logaddexp(0,pre[-1])-target*pre[-1])
            grad=(values[-1]-target)/n
        else:
            loss=np.mean(logsumexp(pre[-1],axis=1)-pre[-1][np.arange(n),y])
            grad=values[-1].copy();grad[np.arange(n),y]-=1;grad/=n
        gw=[None]*len(self.weights);gb=[None]*len(self.weights)
        for i in reversed(range(len(self.weights))):
            gw[i]=values[i].T@grad;gb[i]=grad.sum(axis=0,keepdims=True)
            if i:grad=(grad@self.weights[i].T)*self.derivative(pre[i-1])
        return float(loss),gw,gb

    def predict_proba(self,x):
        p=self.forward(x)[0][-1]
        return np.c_[1-p[:,0],p[:,0]] if self.binary else p

    def fit(self,x,y,xval,yval,epochs=120,lr=.001,seed=42):
        rng=np.random.default_rng(seed)
        parameters=self.weights+self.biases
        m=[np.zeros_like(p) for p in parameters];v=[np.zeros_like(p) for p in parameters]
        step=0;best=float('inf');stale=0;curves=[];saved=None
        for epoch in range(epochs):
            order=rng.permutation(len(x))
            for batch_start in range(0,len(x),32):
                indices=order[batch_start:batch_start+32]
                _,gw,gb=self.loss_gradients(x[indices],y[indices]);grads=gw+gb
                norm=np.sqrt(sum(np.sum(g*g) for g in grads))
                if not np.isfinite(norm):raise ValueError('Pelatihan tidak stabil (NaN/Inf). Periksa data.')
                scale=min(1.,5./max(norm,1e-12));step+=1
                for i,(p,g) in enumerate(zip(parameters,grads)):
                    g=g*scale;m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
                    p-=lr*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
            train_loss=self.loss_gradients(x,y)[0];val_loss=self.loss_gradients(xval,yval)[0]
            curves.append({'epoch':epoch+1,'train_loss':train_loss,'validation_loss':val_loss})
            if val_loss<best-1e-6:
                best=val_loss;stale=0;saved=copy.deepcopy((self.weights,self.biases))
            else:stale+=1
            if stale>=15:break
        if saved:self.weights,self.biases=saved
        return curves

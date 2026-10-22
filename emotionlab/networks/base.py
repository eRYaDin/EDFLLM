"""Jaringan NumPy dari bobot acak; backpropagation + Adam, tanpa model pretrained."""
import copy
import numpy as np
from scipy.special import expit, logsumexp


class Network:
    name = 'base'
    independent_sigmoid = False

    def __init__(self, inputs, classes, hidden=(32,16,8), seed=42):
        if inputs < 1 or classes < 2: raise ValueError('Dimensi model tidak valid.')
        self.inputs, self.classes, self.hidden = inputs, classes, tuple(hidden)
        rng = np.random.default_rng(seed)
        sizes = [inputs,*hidden,classes]
        self.weights = [rng.normal(0,np.sqrt(2/(a+b)),(a,b)) for a,b in zip(sizes[:-1],sizes[1:])]
        self.biases = [np.zeros((1,b)) for b in sizes[1:]]

    def activation(self,z): raise NotImplementedError
    def backward(self,z,gradient): return gradient*self.derivative(z)

    def forward(self,x):
        values,pre = [x],[]
        for i,(w,b) in enumerate(zip(self.weights,self.biases)):
            z=values[-1]@w+b;pre.append(z)
            if i<len(self.weights)-1: a=self.activation(z)
            else: a=expit(z) if self.independent_sigmoid else np.exp(z-logsumexp(z,axis=1,keepdims=True))
            values.append(a)
        return values,pre

    def loss_gradients(self,x,y):
        values,pre=self.forward(x);n=len(x)
        target=np.eye(self.classes)[y]
        if self.independent_sigmoid:
            loss=np.mean(np.sum(np.logaddexp(0,pre[-1])-target*pre[-1],axis=1))
        else: loss=np.mean(logsumexp(pre[-1],axis=1)-pre[-1][np.arange(n),y])
        g=(values[-1]-target)/n;gw=[None]*len(self.weights);gb=[None]*len(self.weights)
        for i in reversed(range(len(self.weights))):
            gw[i]=values[i].T@g;gb[i]=g.sum(axis=0,keepdims=True)
            if i:g=self.backward(pre[i-1],g@self.weights[i].T)
        return float(loss),gw,gb

    def predict_scores(self,x):
        return self.forward(x)[0][-1]

    def fit(self,x,y,xval,yval,epochs=160,lr=.003,seed=42,patience=25):
        rng=np.random.default_rng(seed);params=self.weights+self.biases
        m=[np.zeros_like(p) for p in params];v=[np.zeros_like(p) for p in params]
        step=0;best=float('inf');stale=0;curves=[];saved=None
        for epoch in range(epochs):
            order=rng.permutation(len(x))
            for start in range(0,len(x),32):
                ix=order[start:start+32];_,gw,gb=self.loss_gradients(x[ix],y[ix]);grads=gw+gb
                norm=np.sqrt(sum(np.sum(g*g) for g in grads))
                if not np.isfinite(norm):raise ValueError('Gradien tidak hingga.')
                scale=min(1.,5./max(norm,1e-12));step+=1
                for i,(p,g) in enumerate(zip(params,grads)):
                    g=g*scale;m[i]=.9*m[i]+.1*g;v[i]=.999*v[i]+.001*g*g
                    p-=lr*(m[i]/(1-.9**step))/(np.sqrt(v[i]/(1-.999**step))+1e-8)
            tl=self.loss_gradients(x,y)[0];vl=self.loss_gradients(xval,yval)[0]
            curves.append({'epoch':epoch+1,'train_loss':tl,'validation_loss':vl})
            if vl<best-1e-6:best=vl;stale=0;saved=copy.deepcopy((self.weights,self.biases))
            else:stale+=1
            if stale>=patience:break
        if saved:self.weights,self.biases=saved
        return curves

    def arrays(self):
        return {**{f'w{i}':w for i,w in enumerate(self.weights)},**{f'b{i}':b for i,b in enumerate(self.biases)}}

    def restore(self,arrays):
        for i in range(len(self.weights)):
            w=np.asarray(arrays[f'w{i}']);b=np.asarray(arrays[f'b{i}'])
            if w.shape!=self.weights[i].shape or b.shape!=self.biases[i].shape or not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError('Bobot tidak sesuai arsitektur.')
            self.weights[i]=w.copy();self.biases[i]=b.copy()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 21:02:56 2020

@author: yifeng
"""
import numpy as np
from matplotlib import pyplot as plt

def get_beta(k, a, l, u, T, t):
    """
    Compute the value of beta. When k=0, a is the fixed beta value. Usually we let a=1.
    Special cases:
        when a=0: bate=l
        when k=0: bate=max(a, l)
        when k=0, l=0: bate=a
        when k=0, a=0: bate=b
    INPUTS:
        a, T, t: scalars in formula beta=a*np.exp( k*(1-T/t) ) where a>=0, k>=0.
        l: scalar: l>=0, offset as min value of bate, usually we let l=0.
        u: scalar: l>0. max.
    OUTPUTS:
        bate: scalar.
    """
    bate = a*np.exp( k*(1-T/t) )
    bate = np.max( [bate, l] )
    bate = np.min( [bate, u] )
    return bate

T=100

c1 = [ get_beta(1,1,0,1,T,t) for t in range(1,101)]
c2 = [ get_beta(1,4,0,1,T,t) for t in range(1,101)]
c3 = [ get_beta(1,0.4,0,1,T,t) for t in range(1,101)]
c4 = [ get_beta(1,1,0.01,0.4,T,t) for t in range(1,101)]
c5 = [ get_beta(0,0.1,0,1,T,t) for t in range(1,101)]

x = range(1,101)

# Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
#fig, ax = plt.subplots()  # Create a figure and an axes.
fig=plt.figure(num=1,figsize=(8,4))
ax=fig.add_subplot(1,1,1)
ax.plot(x, c1, label=r'$k=1, a=1, l=0, u=1$') 
ax.plot(x, c2, label=r'$k=4, a=1, l=0, u=1$') 
ax.plot(x, c3, label=r'$k=1, a=0.4, l=0, u=1$') 
ax.plot(x, c4, label=r'$k=1, a=1, l=0.01, u=0.4$')
ax.plot(x, c5, label=r'$k=0, a=0.1, l=0, u=1$')
ax.set_xlabel('Epoch')  # Add an x-label to the axes.
ax.set_ylabel(r'$\beta$')  # Add a y-label to the axes.
#ax.set_title("t")  # Add a title to the axes.
ax.legend()  # Add a legend.
fig.tight_layout()
fig.savefig('./fig_beta.pdf', dpi=300, bbox_inches='tight')


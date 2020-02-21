#functions to create regular polygons

import scipy.misc as sm
import mgcreate as mgc
import scipy.ndimage as nd
import numpy as np
import os
from skimage.draw import line

#create regular n apothemd polygon

def reg_poly(dim=1000, apothem=10, n=3):
    #finding sqrt of dimensionality
    d1 = int(np.sqrt(dim))
    #creating image of zeros
    adata = np.zeros((d1,d1))
    #needed variables
    v_c, h_c = (d1//2, d1//2)
    vals = np.arange(n)
    theta = 360/n
    #finding the corners
    x_corners = np.round(apothem * np.cos(2*np.pi*vals/n + theta) + v_c).astype(int)
    y_corners = np.round(apothem * np.sin(2*np.pi*vals/n + theta) + h_c).astype(int)
    #saving the corners
    adata[x_corners, y_corners] = 1
    #drawing the lines
    rr = []
    cc = []
    for i in range(n-1):
        temp1, temp2 = line(x_corners[i], y_corners[i], x_corners[i+1], y_corners[i+1])
        rr.append (temp1)
        cc.append (temp2)
    temp1, temp2 = line(x_corners[n-1], y_corners[n-1], x_corners[0], y_corners[0])
    rr.append(temp1)
    cc.append(temp2)
    for i in range(n):
        adata[rr[i], cc[i]] = 1
    #fill in the holes
    ultima = nd.binary_fill_holes(adata)
    return(1 - (ultima+0) )




#

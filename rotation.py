import scipy.misc as sm
import mgcreate as mgc
import scipy.ndimage as nd
import numpy as np
import os
import math
import convert as cvt

#convert image to square centered at center of image
#assuming input has the shape as black
def rotations(dir1, fname, deg):
    adata = sm.imread(fname, flatten=True)
    turns = (360.0)/(deg+0.0)
    name = fname[5:]
    for i in range(1, int(turns)):
        temp = nd.rotate(adata, deg*i, reshape=False, cval=255)
        sm.imsave(dir1+"_rot/turn_"+str(i)+"_"+name, temp)


def binary(dir1, fname):
    adata = sm.imread(fname, flatten=True)
    #swapping black and white
    name = fname[9:]
    new = (adata>128) + 0
    sm.imsave(dir1+"_bin/bin_"+name, new)


#obtaining images
def RotateAllImage( dirs ,theta ):
    for d in dirs:
        mgnames = cvt.GetPicNames( d )
        for j in mgnames:
            b = rotations(dirs[0], j,theta )
#

#binarizing images
def BinAllImage( dirs ):
    for d in dirs:
        mgnames = cvt.GetPicNames( d )
        for j in mgnames:
            b = binary(dirs[0], j )


#work
#fname='squs_rot_bin/bin_turn_347_squ_side99.png'
#binary(fname)

#

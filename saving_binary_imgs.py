#saving all binary images

import convert as cvt
import scipy.misc as sm
import mgcreate as mgc
import scipy.ndimage as nd
import numpy as np
import os
import math

#----------------------------------------------------------
#elipse class

elip = ["elip", "none"]

#get all images
imgs = cvt.GetAllImages(elip)
j=1


for i in imgs:
    sm.imsave("elip_bin/"+str(j)+".png",i)
    j += 1


############------------------------------------------------

#square class
spiral = ["spiral", "none"]

#get all images
imgs = cvt.GetAllImages(spiral)
j=1


for i in imgs:
    sm.imsave("spiral_bin/"+str(j)+".png",i)
    j += 1


############------------------------------------------------

#edge class
edge = ["edge_on", "none"]

#get all images
imgs = cvt.GetAllImages(edge)
j=1


for i in imgs:
    sm.imsave("edge_bin/"+str(j)+".png",i)
    j += 1



#


#

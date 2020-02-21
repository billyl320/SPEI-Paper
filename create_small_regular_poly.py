#python program to create lots of hexagons and octagons
import reg_poly as rp
import scipy.misc as sm
import numpy as np

#create n-polygon at range of dimensions
lens = np.arange(1, 126)

#triangle
for i in range(len(lens)):
    temp = rp.reg_poly(dim=100000, apothem=lens[i], n=3)
    sm.imsave('tris/tri_side'+str(lens[i])+'.png', temp)

#square
for i in range(len(lens)):
    temp = rp.reg_poly(dim=100000, apothem=lens[i], n=4)
    sm.imsave('squs/squ_side'+str(lens[i])+'.png', temp)

#pentagon
for i in range(len(lens)):
    temp = rp.reg_poly(dim=100000, apothem=lens[i], n=5)
    sm.imsave('pens/pen_side'+str(lens[i])+'.png', temp)

#hexagon
for i in range(len(lens)):
    temp = rp.reg_poly(dim=100000, apothem=lens[i], n=6)
    sm.imsave('hexs/hex_side'+str(lens[i])+'.png', temp)

#septagon
for i in range(len(lens)):
    temp = rp.reg_poly(dim=100000, apothem=lens[i], n=7)
    sm.imsave('hepts/hept_side'+str(lens[i])+'.png', temp)

#create octagons at range of dimensions
for i in range(len(lens)):
    temp = rp.reg_poly(dim=100000, apothem=lens[i], n=8)
    sm.imsave('octs/oct_side'+str(lens[i])+'.png', temp)

#

import scipy.misc as sm
import mgcreate as mgc
import scipy.ndimage as nd
import scipy.signal as ss
import numpy as np
import skimage as sk
from sklearn import cluster
from skimage import filters
from skimage import feature
from skimage.color import label2rgb
from skimage.feature import corner_harris, corner_peaks
from skimage.measure import regionprops as rp
import os
import math
import colorsys

#convert image to square centered at center of image
#assuming input has the shape as white
def convert(adata):
    #print(fname)
    #swap white to black and black to white
    #adata = (sm.imread(fname, flatten=True) < 125.0) +0.0
    #plopping into square centered at center of image with max of original image
    d = max(adata.shape)*2
    pic = mgc.Plop(adata, (d, d), 0)
    #swapping black and white (if needed [shape needs to be white])
    new = pic
    #new = 1-new
    # find the center of mass
    v = nd.center_of_mass(new)
    v = (int(v[0]), int(v[1]))
    #finding cetner of new image
    n = (d/2, d/2)
    #shift in horizontal
    hori = v[1]-n[1]
    #shift in vertical
    vert = v[0]-n[0]
    #shift the image
    ultima = (nd.shift(new, (-vert, -hori), cval=0) >0.50 )+0.0
    return ultima

#finding minimum encompassing circle - needs to be binary (1 and 0)
def enc_circ(pic):
    ultima = pic + 0
    v,h = ultima.shape
    z = np.ones((v,h))+0
    z[v//2,h//2] = 0
    dist = nd.distance_transform_edt(z)
    vals = dist*ultima
    r = vals.max()
    r = math.ceil(r)
    ultima = ultima[(v//2 - r) : r + v//2, h//2 -r : r + h//2]
    return ultima

#from page 266 of Kinser (2018)
#gives some shape metrics
#this version only provides eccentricity
def Metrics(orig):
    v, h = orig.nonzero()
    mat = np.zeros((2, len(v)))
    mat[0] = v
    mat[1] = h
    evls, evcs = np.linalg.eig(np.cov(mat))
    eccen = evls[0]/evls[1]
    if eccen < 1: eccen = 1/eccen
    return eccen, evls[0], evls[1]

def Shapes(pic):
    #clean image (if needed)
    #pic = nd.gaussian_filter(pic, sigma=1.5)
    #pic = (pic > 0.99) +0.0
    #pic = (nd.binary_erosion(pic , iterations=20))+0.0
    #pic = (nd.binary_dilation(pic , iterations=20))+0.0
    #setup
    shapes = np.zeros((1,5))
    #obtain circularity
    circ = ( sum(sum((nd.binary_dilation(pic , iterations=1) - pic ))) **2) /(4*np.pi*sum(sum(pic)))
    shapes[0][0] = circ
    #provides eccentricity, eigen1 and eigen2
    eccen, e1, e2 = Metrics(pic)
    shapes[0][1] = eccen
    shapes[0][2] = e1
    shapes[0][3] = e2
    #number of corners
    corners = corner_harris(pic, k=0.01)
    shapes[0][4] = corner_peaks(corners, min_distance=1).shape[0]
    #white and black pixel counts for min bounding box
    #theta = rp( (pic>0.5) +0)[0]['orientation']
    #rot_pic = nd.rotate(pic, angle=theta*180/np.pi)
    #rot_pic = (rot_pic>0.5) +0.00
    #slice_x, slice_y = nd.find_objects(rot_pic==1)[0]
    #roi = rot_pic[slice_x, slice_y]
    #shapes[0][5] = np.unique(roi, return_counts=True)[1][1]
    #shapes[0][6] = np.unique(roi, return_counts=True)[1][0]
    return shapes

#gets binary image histogram
def BinaryHist(dirs):
    imgs = []
    imgs.append(GetAllImages(dirs))
    hist = np.zeros( (len(imgs[0]),2) )
    #get histogram values
    for i in range(0,(len(imgs[0]))):
        hist[i][0] = imgs[0][i].sum()
        hist[i][1] = (imgs[0][i].shape[0]*imgs[0][i].shape[1]) - hist[i][0]
    #return vals
    return hist

#getting all image files
#this was changed for the nested folder structure of the data
def GetPicNames( indir ):
    a = os.listdir( indir )
    #remove all files where images begin with a period
    b = [ x for x in a if "._" not in x ]
    #hold all names
    pgmnames= []
    for s in b:
        c = os.listdir(indir+'/'+s)
        d = [ x for x in c if "._" not in x ]
        for t in d:
            if '.tif' in t:
                pgmnames.append( indir+'/'+s + '/' + t )
            if '.png' in t:
                pgmnames.append( indir+'/'+s + '/' + t )
            if '.jpg' in t:
                pgmnames.append( indir+'/'+s + '/' + t )
            if '.bmp' in t:
                pgmnames.append( indir+'/'+s + '/' + t )
    return pgmnames

#obtaining images
def GetAllImages( dirs ):
    mgs = []
    for d in dirs:
        mgnames = GetPicNames( d )
        for j in mgnames:
            print(j)
            adata = (sm.imread(j, flatten=True) < 125.0) +0.0
            b = convert(adata)
            c = enc_circ(b)
            mgs.append( c )
            #90 degrees
            #adata = (sm.imread(j, flatten=True) < 125.0) +0.0
            b = nd.rotate(adata, 90.0, reshape=True, cval=0)
            b = (b>0.50) +0.0
            b = convert(b)
            c = enc_circ(b)
            mgs.append( c )
            #180 degrees
            #adata = (sm.imread(j, flatten=True) < 125.0) +0.0
            b = nd.rotate(adata, 180.0, reshape=True, cval=0)
            b = (b>0.50) +0.0
            b = convert(b)
            c = enc_circ(b)
            mgs.append( c )
            #270 degrees
            #adata = (sm.imread(j, flatten=True) < 125.0) +0.0
            b = nd.rotate(adata, 270.0, reshape=True, cval=0)
            b = (b>0.50) +0.0
            b = convert(b)
            c = enc_circ(b)
            mgs.append( c )
    return mgs

#obtaining images for shape metrics except EIs
def GetAllImages_Shapes( dirs):
    mgs = []
    for d in dirs:
        mgnames = GetPicNames( d )
        for j in mgnames:
            b = sm.imread(j, flatten=True)
            b = (b<125.0) +0.0
            c = Shapes(b)
            mgs.append( c )
            #90 degrees
            b = sm.imread(j, flatten=True)
            b = (b<125.0) +0.0
            b = nd.rotate(b, 90.0, reshape=True, cval=0)
            b = (b>0.50) +0.0
            c = Shapes(b)
            mgs.append( c )
            #180 degrees
            b = sm.imread(j, flatten=True)
            b = (b<125.0) +0.0
            b = nd.rotate(b, 180.0, reshape=True, cval=0)
            b = (b>0.50) +0.0
            c = Shapes(b)
            mgs.append( c )
            #270 degrees
            b = sm.imread(j, flatten=True)
            b = (b<125.0) +0.0
            b = nd.rotate(b, 270.0, reshape=True, cval=0)
            b = (b>0.50) +0.0
            c = Shapes(b)
            mgs.append( c )
    return mgs

#save histogram as .txt where first column is white counts
#and second column is black counts
def BinaryHistTXT(tname, dirs):
    #obtain histogram
    hist = BinaryHist(dirs)
    #get image names
    names = GetPicNames( dirs[0] )
    #save as txt
    np.savetxt(tname, hist, delimiter=',', header="white,black", comments='')
    np.savetxt("NAMES_"+tname, np.asarray(names), delimiter=',', header="image", comments='', fmt="%s")

#save histogram as .txt where first column is white counts
#and second column is black counts
def BinaryShapesTXT(tname, dirs):
    #obtain shape metrics
    shapes = GetAllImages_Shapes(dirs)
    #get image names
    name6 = tname + "_SHAPES.txt"
    #save as txt
    np.savetxt(name6, np.vstack(shapes), delimiter=',', header="Shape_circ, Shape_eccent, Shape_e1, Shape_e2, Shape_corn", comments='')

#

import scipy.misc as sm
import mgcreate as mgc
import scipy.ndimage as nd
import numpy as np
import os
import math

#convert image to square centered at center of image
#assuming input has the shape as black
def convert(fname):
    adata = sm.imread(fname, flatten=True)
    #plopping into square centered at center of image with max of original image
    d = max(adata.shape)*2
    pic = mgc.Plop(adata, (d, d), 256)
    #swapping black and white
    new = (pic>128) + 0
    new = 1 - new
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
    ultima = nd.shift(new, (-vert, -hori), cval=0)
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

#getting all .pgm's and .png's
def GetPicNames( indir ):
    a = os.listdir( indir )
    pgmnames= []
    for t in a:
        if '.pgm' in t:
            pgmnames.append( indir + '/' + t )
        if '.png' in t:
            pgmnames.append( indir + '/' + t )
    return pgmnames

#compute fractal dimension via boxes
#from here https://gist.github.com/viveksck/1110dfca01e4ec2c608515f0d5a5b1d1
#object needs to be white
def fd(fname, threshold=0.9):
    Z = sm.imread(fname, flatten=True)
    # Only for 2d image
    Z = (Z<10)+0.00
    assert(len(Z.shape) == 2)
    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])
    # Transform Z into a binary array
    Z = (Z < threshold)
    # Minimal dimension of image
    p = min(Z.shape)
    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))
    # Extract the exponent
    n = int(np.log(n)/np.log(2))
    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)
    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))
    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]


#obtaining images
def GetAllImages( dirs ):
    mgs = []
    for d in dirs:
        mgnames = GetPicNames( d )
        for j in mgnames:
            b = convert(j)
            c = enc_circ(b)
            mgs.append( c )
    return mgs

#getting all images as is (for CNN)
def GetAllImagesCNN( dirs ):
    mgs = []
    for d in dirs:
        mgnames = GetPicNames( d )
        for j in mgnames:
            adata = sm.imread(j, flatten=True)
            new = (adata>128) + 0
            new = 1 - new
            mgs.append( new )
    return mgs

#histogram conversion for 256 intensities.
def GSHistogram(dirs):
    imgs = []
    imgs.append(GetAllImages(dirs)) #images obtained from directory
    hist = np.zeros( (len(imgs[0]),2) )
    #get histogram values
    for i in range(0,(len(imgs[0]))):
        temp = imgs[0][i].ravel()
        temp = (temp < 256/2) + 0
        hist[i] = np.array(np.histogram(temp, bins=range(0, 3))[0])
    #return vals
    return hist

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

#gets fd of images
def fd_rslt(dirs):
    imgs = []
    imgs.append(GetPicNames(dirs[0]))
    fd_numbs = np.zeros( (len(imgs[0]),1) )
    #get histogram values
    for i in range(0,(len(imgs[0]))):
        fd_numbs[i] = fd( imgs[0][i] )
    #return vals
    return fd_numbs

#save histogram as .txt where first column is white counts
#and second column is black counts
def GSHistTXT(tname, dirs):
    #obtain histogram
    hist = GSHistogram(dirs)
    #save as txt
    np.savetxt(tname, hist, delimiter=',', header="white,black", comments='')

#save histogram as .txt where first column is white counts
#and second column is black counts
def BinaryHistTXT(tname, dirs):
    #obtain histogram
    hist = BinaryHist(dirs)
    #save as txt
    np.savetxt(tname, hist, delimiter=',', header="white,black", comments='')

#save histogram as .txt where first column is white counts
#and second column is black counts
def fdTXT(tname, dirs):
    #obtain histogram
    fd = fd_rslt(dirs)
    #save as txt
    np.savetxt(tname, fd, delimiter=',', header="fd", comments='')

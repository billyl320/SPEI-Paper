#converting images for analysis in R
#importing custom module for analysis
import convert as cvt
import rotation as rot

#desired directories
#note that each class should be separated into different directories.
#however, for the fucntion to work, multiple directories should be specified.
#thus, an empty folder is utilized for this task
#the empty folder is called "none"

#hepts class
hepts = ["hept", "none"]

#converting images
rot.RotateAllImage(hepts, 120.0)

#converting to binary

#square class
hepts = ["hept_rot", "none"]

rot.BinAllImage(hepts)


#converting

#square class
hepts = ["hept_rot_bin", "none"]

#name of .txt file
name = 'hept_rot.txt'

#converting images
cvt.BinaryHistTXT(name, hepts)


############------------------------------------------------

#square class
squs = ["squs", "none"]

#converting images
rot.RotateAllImage(squs, 120.0)

#converting to binary

#square class
squs = ["squs_rot", "none"]

rot.BinAllImage(squs)


#converting

#square class
squs = ["squs_rot_bin", "none"]

#name of .txt file
name = 'squs_rot.txt'

#converting images
cvt.BinaryHistTXT(name, squs)


############------------------------------------------------

#triangle class
tris = ["tris", "none"]

#converting images
rot.RotateAllImage(tris, 120.0)

#converting to binary

#square class
tris = ["tris_rot", "none"]

rot.BinAllImage(tris)


#converting

#square class
tris = ["tris_rot_bin", "none"]

#name of .txt file
name = 'tris_rot.txt'

#converting images
cvt.BinaryHistTXT(name, tris)

############------------------------------------------------

#pentagons class
pens = ["pens", "none"]

#converting images
rot.RotateAllImage(pens, 120.0)

#converting to binary

#square class
pens = ["pens_rot", "none"]

rot.BinAllImage(pens)


#converting

#square class
pens = ["pens_rot_bin", "none"]

#name of .txt file
name = 'pens_rot.txt'

#converting images
cvt.BinaryHistTXT(name, pens)

#----------------------------------------------------------
#hexs class
hexs = ["hexs", "none"]

#converting images
rot.RotateAllImage(hexs, 120.0)

#converting to binary

#square class
hexs = ["hexs_rot", "none"]

rot.BinAllImage(hexs)


#converting

#square class
hexs = ["hexs_rot_bin", "none"]

#name of .txt file
name = 'hexs_rot.txt'

#converting images
cvt.BinaryHistTXT(name, hexs)

#----------------------------------------------------------
#octs class
octs = ["octs", "none"]

#converting images
rot.RotateAllImage(octs, 120.0)

#converting to binary

#square class
octs = ["octs_rot", "none"]

rot.BinAllImage(octs)

#converting

#square class
octs = ["octs_rot_bin", "none"]

#name of .txt file
name = 'octs_rot.txt'

#converting images
cvt.BinaryHistTXT(name, octs)

#

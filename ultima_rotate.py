#converting images for analysis in R
#importing custom module for analysis
import rotation as rot

#desired directories
#note that each class should be separated into different directories.
#however, for the fucntion to work, multiple directories should be specified.
#thus, an empty folder is utilized for this task
#the empty folder is called "none"

#hepts class
hepts = ["hept", "none"]

#converting images
rot.RotateAllImage(hepts, 1.0)

#converting to binary

#square class
hepts = ["hept_rot", "none"]

rot.BinAllImage(hepts)


############------------------------------------------------

#square class
squs = ["squs", "none"]

#converting images
rot.RotateAllImage(squs, 1.0)

#converting to binary

#square class
squs = ["squs_rot", "none"]

rot.BinAllImage(squs)


############------------------------------------------------

#triangle class
tris = ["tris", "none"]

#converting images
rot.RotateAllImage(tris, 1.0)

#converting to binary

#square class
tris = ["tris_rot", "none"]

rot.BinAllImage(tris)

############------------------------------------------------

#pentagons class
pens = ["pens", "none"]

#converting images
rot.RotateAllImage(pens, 1.0)

#converting to binary

#square class
pens = ["pens_rot", "none"]

rot.BinAllImage(pens)

#----------------------------------------------------------
#hexs class
hexs = ["hexs", "none"]

#converting images
rot.RotateAllImage(hexs, 1.0)

#converting to binary

#square class
hexs = ["hexs_rot", "none"]

rot.BinAllImage(hexs)

#----------------------------------------------------------
#octs class
octs = ["octs", "none"]

#converting images
rot.RotateAllImage(octs, 1.0)

#converting to binary

#square class
octs = ["octs_rot", "none"]

rot.BinAllImage(octs)
#

#converting images for analysis in R
#importing custom module for analysis
import convert as cvt

#desired directories
#note that each class should be separated into different directories.
#however, for the fucntion to work, multiple directories should be specified.
#thus, an empty folder is utilized for this task
#the empty folder is called "none"

#----------------------------------------------------------
#elipse class

elip = ["elip", "none"]

#name of .txt file
name = 'elip.txt'

#converting images
cvt.BinaryHistTXT(name, elip)



#name of .txt file
name2 = 'elip_old.txt'

#converting images
cvt.BinaryHistTXT_old(name2, elip)

############------------------------------------------------

#square class
spiral = ["spiral", "none"]

#name of .txt file
name = 'spiral.txt'

#converting images
cvt.BinaryHistTXT(name, spiral)

#name of .txt file
name2 = 'spiral_old.txt'

#converting images
cvt.BinaryHistTXT_old(name2, spiral)

############------------------------------------------------

#edge class
edge = ["edge_on", "none"]

#name of .txt file
name = 'edge.txt'

#converting images
cvt.BinaryHistTXT(name, edge)

#name of .txt file
name2 = 'edge_old.txt'

#converting images
cvt.BinaryHistTXT_old(name2, edge)



#

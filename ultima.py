#converting images for analysis in R
#importing custom module for analysis
import convert as cvt

#desired directories
#note that each class should be separated into different directories.
#however, for the fucntion to work, multiple directories should be specified.
#thus, an empty folder is utilized for this task
#the empty folder is called "none"

#hepts class
hepts = ["hepts", "none"]

#name of .txt file
name = 'hepts.txt'

#converting images
cvt.BinaryHistTXT(name, hepts)

#name of .txt file
name2 = 'hepts_old.txt'

#converting images
cvt.BinaryHistTXT_old(name2, hepts)

#----------------------------------------------------------
#decs class
decs = ["decs", "none"]

#name of .txt file
name = 'decs.txt'

#converting images
cvt.BinaryHistTXT(name, decs)

#name of .txt file
name2 = 'decs_old.txt'

#converting images
cvt.BinaryHistTXT_old(name2, decs)

############------------------------------------------------

#enns class
enns = ["enns", "none"]

#name of .txt file
name = 'enns.txt'

#converting images
cvt.BinaryHistTXT(name, enns)

#name of .txt file
name2 = 'enns_old.txt'

#converting images
cvt.BinaryHistTXT_old(name2, enns)

############------------------------------------------------

#square class
squs = ["squs", "none"]

#name of .txt file
name = 'squs.txt'

#converting images
cvt.BinaryHistTXT(name, squs)

#name of .txt file
name2 = 'squs_old.txt'

#converting images
cvt.BinaryHistTXT_old(name2, squs)

############------------------------------------------------

#triangle class
tris = ["tris", "none"]

#name of .txt file
name = 'tris.txt'

#converting images
cvt.BinaryHistTXT(name, tris)

#name of .txt file
name2 = 'tris_old.txt'

#converting images
cvt.BinaryHistTXT_old(name2, tris)

############------------------------------------------------

#pentagons class
pens = ["pens", "none"]

#name of .txt file
name = 'pens.txt'

#converting images
cvt.BinaryHistTXT(name, pens)

#name of .txt file
name2 = 'pens_old.txt'

#converting images
cvt.BinaryHistTXT_old(name2, pens)

#----------------------------------------------------------
#hexs class
hexs = ["hexs", "none"]

#name of .txt file
name = 'hexs.txt'

#converting images
cvt.BinaryHistTXT(name, hexs)

#name of .txt file
name2 = 'hexs_old.txt'

#converting images
cvt.BinaryHistTXT_old(name2, hexs)

#----------------------------------------------------------
#octs class
octs = ["octs", "none"]

#name of .txt file
name = 'octs.txt'

#converting images
cvt.BinaryHistTXT(name, octs)

#name of .txt file
name2 = 'octs_old.txt'

#converting images
cvt.BinaryHistTXT_old(name2, octs)

#

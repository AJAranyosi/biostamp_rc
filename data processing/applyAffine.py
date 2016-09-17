#!/usr/bin/env python
'''
Usage: applyAffine.py affine.csv accel.csv output.csv
  applies the affine transformation stored in csv format in affine.csv
  to the data in accel.csv. The affine transform in 3D is defined by

  [ x_cal ]   [ a_xx a_xy a_xz a_xc ] [ x_meas ]
  [ y_cal ] = [ a_yx a_yy a_yz a_yc ] [ y_meas ]
  [ z_cal ]   [ a_zx a_zy a_zz a_zc ] [ z_meas ]
  [   1   ]   [  0    0    0    1   ] [    1   ]

  where {x,y,z}_cal are the transformed outputs, and {x,y,z}_meas are
  the input values from accel.csv. This script should work with
  transforms of other dimensions, but hasn't been tested.
  Output data is written into output.csv. Note that the first column
  is assumed to be timestamps, and is copied over unchanged. Also the
  first row is copied unchanged.
'''

def loadAffine(affineName):
    '''Reads the affine transform from a csv file.'''
    affine= []
    for line in open(affineName):
        affine.append([float(x) for x in line.strip().split(',')])
    return affine

def affineXform(data, affine):
    data.append(1)
    xformed= [sum([row[i] * data[i] for i in range(len(row))]) for row in affine]
    return(xformed)

def applyAffine(inName, outName, affine):
    '''Applies the affine transform to inName, writing
       the result to outName'''
    outFile= open(outName, 'w')
    inFile= open(inName)
    outFile.write(inFile.readline()) # Copy headers unchanged
    for line in inFile:
        fields= line.strip().split(',')
        xformed= affineXform([float(x) for x in fields[1:]], affine)
        outFile.write('{}\n'.format(','.join([fields[0]] + [str(x) for x in xformed])))
    outFile.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print(__doc__)
        exit()

    affine= loadAffine(sys.argv[1])
    applyAffine(sys.argv[2], sys.argv[3], affine)


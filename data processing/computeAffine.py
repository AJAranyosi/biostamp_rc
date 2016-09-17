#!/usr/bin/env python
'''
Usage: computeAffine.py annotations.csv accel.csv
  annotations.csv should contain Activities with answers to associated questions.
  the answers should be one each of "+x", "-x", "+y", "-y", "+z", "-z", corresponding
  to the orientation of the calibration box during each performance of the Activity.
  This script will then extract the corresponding time series data from the accel.csv
  file and compute the average acceleration in the x, y, and z directions for each
  Activity. These accelerations will be used to derive an affine transform that can
  be used to calibrate accelerometer data from this Sensor using a y=Ax equation:
  
  [ x_cal ]   [ a_xx a_xy a_xz a_xc ] [ x_meas ]
  [ y_cal ] = [ a_yx a_yy a_yz a_yc ] [ y_meas ]
  [ z_cal ]   [ a_zx a_zy a_zz a_zc ] [ z_meas ]
  [   1   ]   [  0    0    0    1   ] [    1   ]

  where {x,y,z}_cal are the calibrated values, and {x,y,z}_meas are the values reported
  by the accelerometer.

  This script outputs the A matrix for this equation. It also writes this matrix to the
  file affine.csv. A separate script, applyAffine.csv, can apply this transformation to
  an existing data set to recover the calibrated values.
'''

import math
import numpy as np
from numpy.linalg import lstsq

def getCalibrationTimestamps(annoName, startTs, stopTs):
    '''Get the timestamps of the Activities associated with each calibration measurement.
       The process involves finding all of the Activities that were performed between
       startTs and stopTs, finding all of the Questions that were answered after Activities,
       associating each Question with its corresponding Activity, then extracting the subset
       that correspond to the calibration measurements (i.e., Activities where the questions
       were answered with +x, -x, +y, -y, +z, or -z).
    '''
    annos= {}  # Holds the annotations associated with each Activity
    labels= {} # Holds the annotations associated with each Question answered after an Activity
    annoFile= open(annoName)
    annoFile.readline() # Skip header line
    # Extract the Activities and Questions from the annotations file
    for line in annoFile:
        fields= line.strip().split(',')
        if fields[6] == '' and 'Activity' in fields[2]:
            if float(fields[4]) >= startTs and float(fields[5]) <= stopTs:
                annos[fields[1]]= [float(fields[4]), float(fields[5]), ''] # Capture timestamps, leave space for annotation
        elif 'Activity' in fields[2] and 'Question' in fields[2]:
            if float(fields[4]) >= startTs and float(fields[5]) <= stopTs:
                labels[fields[1]]= [fields[6].lower(), float(fields[5])] # Capture the labels and timestamps for question answers

    # Associate each Question answer with the corresponding Activity - not trivial, so we have to look at timestamps
    # Sort from last to first, and make a copy so Python doesn't get confused as we change the original
    sortedAnnos= sorted(annos, key= lambda x: annos[x][-2], reverse= True)[:]
    sortedLabels= sorted(labels, key= lambda x: labels[x][-1])
    for label in sortedLabels:
        for anno in sortedAnnos:
            if annos[anno][-2] <= labels[label][-1]: # Once we find the first anno whose stop time is before the label time
                annos[anno][-1]= labels[label][0] # Add the label to the annotation and break
                break;

    # Finally, pick out the Calibration annotations and make sure we have the right number
    relevantAnnos= [annos[anno][-3:] for anno in annos if annos[anno][-1] in ['+x', '+y', '+z', '-x', '-y', '-z']]
    if len(relevantAnnos) == 6:
        return relevantAnnos
    print("Can't find the right set of annotations in {}! Got {}".format(annoFile, repr([anno for anno in annos])))
    return None

def getCalibrationMeanStdDev(dataName, startTs, stopTs):
    '''Given a CSV file and a start and stop timestamp, where the first column corresponds to timestamps,
       return a list where each entry contains the mean and stdev of values in the corresponding data
       column between the start and stop timestamps. If no values are found, return 0 for each column.
    '''
    dataFile= open(dataName)
    line= dataFile.readline() # Skip header line
    # Make sure it's at least a 4-column CSV
    fields= line.strip().strip(',').split(',') # Remove CR and any trailing commas
    nvals= len(fields) - 1
    vals= np.empty((0, nvals))
    for line in dataFile:
        fields= line.strip().strip(',').split(',')
        if startTs <= float(fields[0]) <= stopTs:
            vals= np.append(vals, [[float(x) for x in fields[1:] if x is not '']], 0)
        # No need to continue once we've gone beyond stopTs
        if float(fields[0]) > stopTs:
            break
    if len(vals) > 0:
        valsums= vals.sum(axis= 0)
        valsumsqs= (vals*vals).sum(axis= 0)
        n= len(vals)
        means= valsums / n
        stdevs= [math.sqrt(x) for x in (valsumsqs - valsums * valsums / n) / (n - 1)] # Sample std dev
        return([x for x in zip(list(means), stdevs)])
    else:
        return([x for x in zip([0] * nvals, [0] * nvals)])

def getStartStopTimestamps(dataName):
    '''Get the initial and final timestamp from a data file.
    '''
    dataFile= open(dataName)
    dataFile.readline() # Skip header
    fields= dataFile.readline().strip().split(',')
    startTs= float(fields[0])
    for line in dataFile:
        pass
    fields= line.strip().split(',')
    stopTs= float(fields[0])
    return [startTs, stopTs]

def deriveAffine(msds, axes):
    '''Derive the Affine transform for converting measured data to calibrated data. This method assumes that
       we have +n and -n gravity calibrations for each axis n. msds is a dict where the keys are '+n' and '-n'
       for each axis n, and the values are a list of length 2, where the first item is a list of length m
       (m being the number of axes) corresponding to the average values recorded from each axis during the
       calibration. The second item in the list contains standard deviations, and is ignored in this calculation.
       axes is an ordered list of axis names, where the order corresponds to the ordering in the data (e.g.,
       ['x', 'y', 'z']).
       The approach taken is to rewrite the equation at the top of this file in the following form:
       Xmeas * a = xcal', where for a 3D transform,
         Xmeas = [x_meas y_meas z_meas 1 0 0 0 0 0 0 0 0;
                  0 0 0 0 x_meas y_meas z_meas 1 0 0 0 0;
                  0 0 0 0 0 0 0 0 x_meas y_meas z_meas 1;
                  ...]
       for each entry in msds,
         a= [a_xx a_xy a_xz a_xc a_yx a_yy a_yz a_yc a_zx a_zy a_zz a_zc]',
       and
         xcal= [1 0 0 -1 0 0 0 1 0 0 -1 0 0 0 1 0 0 -1]'
       corresponding to the 'correct' values for the 6 calibration measurements.
       Then our least-squares estimate of a is given by
         a= xcal \ Xmeas
    '''
    naxes= len(axes)
    # our Xmeas array will be of dimension (naxes * naxes * 2, (naxes + 1) * naxes);
    # for each direction of each axis, we'll have naxes rows of values; each row
    # will have naxes + 1 values for each axis.
    Xmeas= np.empty((0, (naxes + 1) * naxes))
    xcal= np.empty((0, 1))
    filler= [0] * (naxes + 1) # Used to pad the relevant values for each row of Xmeas
    for axis in range(len(axes)):
        for orientation in ['+', '-']:
            # Extract calibration data from msds for the given axis and orientation
            data= msds['{}{}'.format(orientation, axes[axis])]
            vals= []
            for valaxis in range(len(axes)):
                vals.append(data[valaxis][0])
                # Add the appropriate calibration value to xcal
                if valaxis == axis:
                    if '-' in orientation:
                        xcal= np.append(xcal, [[-1]], 0)
                    else:
                        xcal= np.append(xcal, [[1]], 0)
                else:
                    xcal= np.append(xcal, [[0]], 0)
            vals.append(1.0)
            for eqaxis in range(len(axes)):
                Xmeas= np.append(Xmeas, [filler * eqaxis + vals + filler * (len(axes) - 1 - eqaxis)], 0)
    a= lstsq(Xmeas, xcal)[0].reshape((naxes, naxes + 1))
    return(a)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(__doc__)
        exit()

    [startTs, stopTs]= getStartStopTimestamps(sys.argv[2])
    annos= getCalibrationTimestamps(sys.argv[1], startTs, stopTs)
    msds= {} # Means and std devs
    for anno in annos:
        # Scan through the accel file for each entry and capture the corresponding msds
        msds[anno[-1]]= getCalibrationMeanStdDev(sys.argv[2], anno[0], anno[1])

    # Print out some simple statistics about the accelerometer performance - this ignores skew and other issues
    axes= ['x', 'y', 'z']
    print()
    print("Calibration measurements:")
    for axis in range(len(axes)): # Specify explicitly to ensure ordering
        gain= (msds['+{}'.format(axes[axis])][axis][0] - msds['-{}'.format(axes[axis])][axis][0]) / 2
        offset= (msds['+{}'.format(axes[axis])][axis][0] + msds['-{}'.format(axes[axis])][axis][0]) / 2
        stdev= offset= (msds['+{}'.format(axes[axis])][axis][1] + msds['-{}'.format(axes[axis])][axis][1]) / 2
        print("{}: Gain: {: 0.6f} Offset: {: 0.8f} CV: {: 0.8f}".format(axes[axis], gain, offset, stdev / gain)) # Should print 6 significant figures for all values, assuming offset and CV are <0.01

    affine= deriveAffine(msds, axes)
    print()
    print("Affine Transform:")
    print("[ x_cal ]   [{: 0.6f} {: 0.6f} {: 0.6f} {: 0.6f}] [ x_meas ]".format(affine[0][0], affine[0][1], affine[0][2], affine[0][3]))
    print("[ y_cal ] = [{: 0.6f} {: 0.6f} {: 0.6f} {: 0.6f}] [ y_meas ]".format(affine[1][0], affine[1][1], affine[1][2], affine[1][3]))
    print("[ z_cal ]   [{: 0.6f} {: 0.6f} {: 0.6f} {: 0.6f}] [ z_meas ]".format(affine[2][0], affine[2][1], affine[2][2], affine[2][3]))
    print("[   1   ]   [ 0.000000  0.000000  0.000000  1.000000] [    1   ]")

    outFile= open('affine.csv', 'w')
    for row in affine:
        outFile.write('{}\n'.format(','.join([str(x) for x in list(row)])))
    outFile.close()
    print()
    print("Transform written to affine.csv")
    
    

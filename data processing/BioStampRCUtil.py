#!/usr/bin/env python

'''BioStampRCUtil.py - utility code for interfacing with BioStampRC data
'''

import math			# Used for meanStdDev
import numpy as np  # Data for reading and writing is assumed to be a numpy array

def meanStdDev(values):
    '''Compute the mean and standard deviation of an iterable'''
    if len(values) > 0:
        datasum= sum(values)
        datasumsq= sum([x * x for x in values])
        n= len(values)
        mean= datasum / n
        stdev= math.sqrt((datasumsq - datasum * datasum / n) / (n - 1))
        return([mean, stdev])
    else:
        return([0, 0])

##########################
### File I/O Functions ###
##########################

def readDataFile(filename, headerLines= 1):
    '''Read data from a BioStampRC CSV file. Requires a filename as argument.
    Returns data, timestamps, header in that order. Header is a string representing
    the first row of the file (the optional argument headerLines allows you to specify
    the number of rows of header information). Timestamps is a list of the remaining values in the first
    column. Data is a 2D numpy array representing the remaining columns of data.'''
    if filename is None:
        return False
    file= open(filename)
    header= ""
    for line in range(headerLines):
        header+= file.readline()
    timestamps= []
    data= [] # Create as a list of lists, return as a numpy array
    for line in file:
        fields= line.strip().split(',')
        timestamps.append(float(fields[0]))
        data.append([float(x) for x in fields[1:]])
    file.close()
    npData= np.asarray(data)
    return npData, timestamps, header

def writeDataFile(filename, header, timestamps, data):
    '''Write output data to the specified file. Requires filename, header, timestamps, data
    as arguments. Filename is the full name and path of the file to write to. Header is a
    string written at the top of the file. Timestamps is a list of the first column of data.
    Data is a numpy array of the remaining columns.'''
    if filename is None:
      	return False
    file= open(filename, 'w')
    file.write(header)
    # Note: the * in the following line converts the outermost list into individual arguments to zip()
    for fields in zip(timestamps, *[data[:,i] for i in range(data.shape[1])]): # Add timestamps to an arbitrary number of data columns
        file.write("{}\n".format(",".join([str(x) for x in fields]))) # Convert list to CSV form and write
    file.close()

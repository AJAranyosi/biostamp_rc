#!/usr/bin/env python
'''Usage: BioStampRCDataFilter.py infile outfile low_cutoff high_cutoff
   Provides examples of using the dataFilter class to filter BioStampRC data.
   Filtered data is written to files named outfile_filterType.csv.
   Multiple filters are used, as described in the code at the bottom of the file.
   Filter works on either biopotential (2 column) or IMU (4 or 7 column) CSV files.
   The sample rate is computed automatically from the timestamps in the first column.
   low_cutoff should be the lower bound of the cutoff frequency (e.g. 10)
   high_cutoff should be the upper bound of the cutoff frequency (e.g. 30)
   If low_cutoff = 0, implement a low-pass rather than bandpass filter
   If high_cutoff = 0, implement a high-pass rather than bandpass filter.
'''

import numpy as np
from scipy.signal import butter, lfilter, iirdesign
from BioStampRCUtil import meanStdDev, readDataFile, writeDataFile
import math

class BioStampRCDataFilter(object):
    '''A class for filtering data collected and exported by BioStampRC. The type of filter,
    filter parameters, and data set can all be set and manipulated separately, allowing you
    to apply different filters to the same data set, or the same filter to multiple data sets.
    The default filter is an IIR band-pass filter with a pass band of 5-40 Hz, which is suitable
    for recordings of electro-cardiac activity. A high-pass filter with a cutoff frequency of
    25 Hz is more suitable for electrical recordings from other muscles. Various functions within
    this class provide precise control over the data, filter, and output.
    '''

    def __init__(self, lowFreq= 5, highFreq= 40, band="bandpass", filtType="iir", order= 5, lowStop= None, highStop= None, minStopAttenuation= 40, maxPassRipple= 3, medfiltLength= 25, header= None, timestamps= None, data= None, dataInputFileName= None, dataOutputFileName= None):
        self.filterBands= ["lowpass", "highpass", "bandpass"]
        self.filterTypes= ["fir", "iir", "notch", "median"]
        self.setFilterParams(lowFreq, highFreq, band, filtType, order, lowStop, highStop, minStopAttenuation, maxPassRipple, medfiltLength)
        if header is not None and timestamps is not None and data is not None:
            self.setData(header, timestamps, data)
        elif dataInputFileName is not None:
            self.getDataFromFile(dataInputFileName)
        if dataOutputFileName is not None:
            self.dataOutputFileName= dataOutputFileName

    #################################################
    ### Data assignment and calculation functions ###
    #################################################

    def getDataFromFile(self, filename= None):
        if filename is not None:
            self.dataInputFileName= filename
        data, timestamps, header= readDataFile(self.dataInputFileName)
        self.setData(header, timestamps, data)

    def writeDataToFile(self, filename= None):
        if filename is not None:
            self.dataOutputFileName= filename
        writeDataFile(self.dataOutputFileName, self.header, self.timestamps, self.filtData)

    def setData(self, header, timestamps, data):
        '''Set data from iterables'''
        self.header= header
        self.timestamps= timestamps
        self.data= np.asarray(data) # Make sure we convert it to a numpy array
        self.filtData= np.zeros(self.data.shape) # Create filtData so it's not empty
        self.computeFs()

    def computeFs(self):
        '''Compute average sampling frequency from timestamps'''
        if self.timestamps is None:
            self.fs= 1
            return
        [mean, stdev]= meanStdDev(self.getIntervals())
        self.fs= 1000.0 / mean

    def getIntervals(self):
        '''Compute intervals between timestamps. Used to get actual sampling rate.'''
        return [y - x for (x,y) in zip(self.timestamps[:-1], self.timestamps[1:])]

    ###############################
    ### Filter Design Functions ###
    ###############################

    # Note: the IIR filter is almost certainly better than the Butterworth, so use that by default.
    # Butterworth is good for doing real-time filtering, but this class isn't really designed for that.
    # If we want to handle sample rate variability more gracefully, we might be able to do that
    # with the Butterworth

    def butterworthFilter(self):
        '''Design a butterworth FIR filter'''
        okay, reason= self.sanityCheck()
        if okay is False:
            print("Sanity check failed: {}".format(reason))
            return okay, reason
        nyq= 0.5 * self.fs
        low= self.lowFreq / nyq
        high= self.highFreq / nyq
        if self.band == "lowpass" or self.lowFreq == 0:
            b, a= butter(self.order, high, btype='lowpass')
        elif self.band == "highpass" or self.highFreq == 0:
            b, a= butter(self.order, low, btype='highpass')
        else:
            b, a= butter(self.order, [low, high], btype='bandpass')
        return b, a

    def ellipticalFilter(self):
        '''Design an elliptical IIR filter'''
        okay, reason= self.sanityCheck()
        if okay is False:
            print("Sanity check failed: {}".format(reason))
            return okay, reason
        nyq= 0.5 * self.fs
        if self.band == "lowpass" or self.lowFreq == 0: # Lowpass filter - highStop must be > highFreq
            wp= self.highFreq / nyq
            ws= self.highStop / nyq
        elif self.band == "highpass" or self.highFreq == 0: # Highpass - lowStop must be < lowFreq
            wp= self.lowFreq / nyq
            ws= self.lowStop / nyq
        elif self.band == "bandpass": # Bandpass: lowStop must be < lowFreq, highStop must be > highFreq, lowFreq must be < highFreq
            wp= [self.lowFreq / nyq, self.highFreq / nyq]
            ws= [self.lowStop / nyq, self.highStop / nyq]
        if self.minStopAttenuation is None:
            gstop= 6 * self.order # minimum attenuation in the stop band - 6 dB for each "filter order"
        else:
            gstop= self.minStopAttenuation
        if self.maxPassRipple is None:
            gpass= 3 # max attenuation in the pass band
        else:
            gpass= self.maxPassRipple
        ftype= 'ellip'
        return iirdesign(wp= wp, ws= ws, gstop= gstop, gpass= gpass, ftype= ftype)

    def notchFilter(self):
        '''Design a notch filter. Notch is at frequency self.lowFreq. Bandwidth is 2 * abs(self.lowFreq - self.lowStop).
        Lifted from Sturla Molden-2's comment at http://scipy-user.10969.n7.nabble.com/Very-simple-IIR-filters-td13296.html'''
        okay, reason= self.sanityCheck()
        if okay is False:
            print("Sanity check failed: {}".format(reason))
            return okay, reason
        nyq= 0.5 * self.fs
        ws= self.lowFreq / nyq / 2.0
        bandwidth= 2.0 * abs(self.lowFreq - self.lowStop) / nyq
        R= 1.0 - 3.0 * (bandwidth / 2.0)
        K= ((1.0 - 2.0 * R * np.cos(2 * np.pi * ws) + R**2) / (2.0 - 2.0 * np.cos(2 * np.pi * ws)))
        b, a= np.zeros(3), np.zeros(3)
        a[0], a[1], a[2]= 1.0, -2.0 * R * np.cos(2 * np.pi * ws), R**2
        b[0], b[1], b[2]= K, -2.0 * K * np.cos(2 * np.pi * ws), K
        return b, a

    def medianFilter(self, x):
        """Apply a median filter to a 1D array x.
        Length is specified by self.medfiltLength
        Boundaries are extended by repeating endpoints.
        """
        okay, reason= self.sanityCheck()
        if okay is False:
            print("Sanity check failed: {}".format(reason))
            return okay, reason
        assert x.ndim == 1, "Input must be one-dimensional."
        k2 = (self.medfiltLength - 1) // 2
        y = np.zeros ((len (x), self.medfiltLength), dtype=x.dtype)
        y[:,k2] = x
        for i in range (k2):
            j = k2 - i
            y[j:,i] = x[:-j]
            y[:j,i] = x[0]
            y[:-j,-(i+1)] = x[j:]
            y[-j:,-(i+1)] = x[-1]
        return np.median (y, axis=1)

    def sanityCheck(self):
        '''Check that various filter parameters are sensible.'''
        # Don't check whether we have data here - we may design filter before loading data
        okay= True # Set to False on problems
        reason= "All good!"
        if not self.filtType in self.filterTypes:
            okay= False
            reason= "Filter type {} not in valid list {}".format(self.filtType, self.filterTypes)
        if "median" in self.filtType:
            if self.medfiltLength % 2 != 1:
                okay= False
                reason= "Median filter length {} must be odd.".format(self.medfiltLength)
            return okay, reason # Don't test anything else if we're median filtering
        if not self.band in self.filterBands:
            okay= False
            reason= "Band type {} not in valid list {}".format(self.band, self.filterBands)
        if not 0 <= self.lowFreq < self.fs / 2.0:
            okay= False
            reason= "lowFreq {} not in valid range".format(self.lowFreq)
        if not 0 <= self.highFreq < self.fs / 2.0: 
            okay= False
            reason= "highFreq {} not in valid range".format(self.highFreq)
        if 0 < self.highFreq <= self.lowFreq:
            okay= False
            reason= "highFreq {} below lowFreq {} but not 0".format(self.highFreq, self.lowFreq)
        if self.highFreq == 0 and self.lowFreq == 0: 
            okay= False
            reason= "both lowFreq and highFreq are 0"
        if 'iir' in self.filtType and not 0 <= self.lowStop < self.fs / 2.0: 
            okay= False
            reason= "lowStop {} not in valid range".format(self.lowStop)
        if 'iir' in self.filtType and not 0 <= self.highStop < self.fs / 2.0:
            okay= False
            reason= "highStop {} not in valid range".format(self.highStop)
        if 'iir' in self.filtType and self.lowStop != 0 and not 0 < self.lowStop <= self.lowFreq:
            okay= False
            reason= "lowStop {} not below lowFreq {}".format(self.lowStop, self.lowFreq)
        if 'iir' in self.filtType and self.highStop != 0 and 0 < self.highStop <= self.highFreq:
            okay= False
            reason= "highStop {} not above highFreq {}".format(self.highStop, self.highFreq)
        if 'iir' in self.filtType and self.highStop != 0 and 0 < self.highStop < self.lowStop:
            okay= False
            reason= "highStop {} not above lowStop {}".format(self.highStop, self.lowStop)
        return okay, reason

    def setFilterParams(self, lowFreq= None, highFreq= None, band= None, filtType= None, order= None, lowStop= None, highStop= None, minStopAttenuation= None, maxPassRipple= None, medfiltLength= None):
        if lowFreq is not None:
            self.lowFreq= lowFreq       # Low frequency cutoff. Ignored for low-pass filters. Set to 0 to explicitly specify low-pass operation.
        if highFreq is not None:
            self.highFreq= highFreq     # High frequency cutoff. Ignored for high-pass filters. Set to 0 to explicity specify high-pass operation.
        if band in self.filterBands:
            self.band= band             # Specify "lowpass", "bandpass", or "highpass". Can be used in place of setting lowFreq or highFreq to 0.
        if filtType in self.filterTypes:
            self.filtType= filtType     # Specify "iir" for an elliptical IIR filter, or "fir" for a Butterworth FIR filter.
        if order is not None:
            self.order= order           # Set the filter order for FIR filter. for IIR filter, this value is multiplied by 6 to determine the stop-band attenuation in dB.
        if lowStop is None:         # For IIR band-pass and high-pass filters, frequencies below lowStop will be attenuated by at least minStopAttenuation dB.
            self.lowStop= 0.75 * self.lowFreq
        else:
            self.lowStop= lowStop
        if highStop is None:        # For IIR low-pass and band-pass filters, frequencies above highStop will be attenuated by at least minStopAttenuation dB.
            self.highStop= 1.1 * self.highFreq 
        else:
            self.highStop= highStop
        if minStopAttenuation is not None:
            self.minStopAttenuation= minStopAttenuation # Minimum attenuation in the stop band for IIR filter, in dB
        if maxPassRipple is not None:
            self.maxPassRipple= maxPassRipple           # Maximum ripple in the pass band for IIR filter, in dB
        if medfiltLength is not None:
            self.medfiltLength= medfiltLength

    def filterData(self):
        '''Filter each column of data, using the filter as currently designed'''
        if self.data is None:
            print("No data to filter!")
            return False
        if self.filtType is 'iir':
            b, a= self.ellipticalFilter()
        elif self.filtType is 'fir':
            b, a= self.butterworthFilter()
        elif self.filtType is 'notch':
            b, a= self.notchFilter()
        elif self.filtType is 'median':
            b, a= True, True
        else:
            print("Filter type {} not recognized! Not filtering data.".format(self.filtType))
            return False
        if b is False: # Failed sanity check
            print("Can't design filter!")
            return False
        self.filtData= np.zeros(self.data.shape)
        for col in range(self.data.shape[1]):
            if "median" in self.filtType:
                self.filtData[:,col]= self.medianFilter(self.data[:,col])
            else:
                self.filtData[:,col]= lfilter(b, a, self.data[:,col])

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 5:
        print(__doc__)
        exit()
    inName= sys.argv[1]
    outName= sys.argv[2]
    lowFreq= float(sys.argv[3])
    highFreq= float(sys.argv[4])

    # First example: initialize default filter, then step through the process, giving minimal arguments
    print("Example 1: IIR Filter")
    # To use this code outside the BioStampRCDataFilter.py file, uncomment the following line
    # from BioStampRCDataFilter import BioStampRCDataFilter
    filter= BioStampRCDataFilter() # We only need to create this once
    filter.getDataFromFile(inName) # We read data in the first time, then apply various filters to it in the examples
    filter.setFilterParams(lowFreq= lowFreq, highFreq= highFreq) # Leave all other filter parameters as default
    filter.filterData()
    filter.writeDataToFile(outName + "_iirFilter.csv")

    # Second example: modify filter by changing a few parameters - others remain unchanged.
    print("Example 2: IIR Filter with adjusted parameters")
    filter.setFilterParams(lowStop= 0.8 * lowFreq, highStop= 1.25 * highFreq, minStopAttenuation= 60, maxPassRipple= 2)
    filter.filterData()
    filter.writeDataToFile(outName + "_iirFilter2.csv")
    # Save the data for use in example 4
    header= filter.header
    timestamps= filter.timestamps[:] # Make a new copy so it doesn't get changed later
    data= filter.filtData.copy()  # Copy the filtered data

    # Third example: use FIR filter instead
    print("Example 3: FIR Filter")
    filter.setFilterParams(filtType= "fir", order= 5)
    filter.filterData()
    filter.writeDataToFile(outName + "_firFilter.csv")

    # Fourth example: use the same filter, but run it on the filtered data we saved from the second example
    print("Example 4: FIR filter on previously filtered data")
    filter.setData(header, timestamps, data) # Alternate method: filter.getDataFromFile(outName + "_iirFilter2.csv")
    filter.filterData()
    filter.writeDataToFile(outName + "_iir+firFilter.csv")

    # Apply notch filter to data
    print("Example 5: Notch Filter")
    filter= BioStampRCDataFilter() # We only need to create this once
    filter.getDataFromFile(inName) # We read data in the first time, then apply various filters to it in the examples
    filter.setFilterParams(lowFreq= lowFreq, highFreq= highFreq, lowStop= 0.95 * lowFreq, filtType= "notch") # Leave all other filter parameters as default
    filter.filterData()
    filter.writeDataToFile(outName + "_notchFilter.csv")

    # Apply a 25-point median filter to data
    print("Example 6: Median Filter")
    filter.setFilterParams(filtType= "median", medfiltLength= 25)
    filter.filterData()
    filter.writeDataToFile(outName + "_medianFilter.csv")

# Import libraries
import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset

class DataHandler:
    def __init__(self, data_directory: str):
        """
        Initialize the DataHandler class with the data directory path. 

        :param data_directory: Path to the .mseed file

        """
        self.data_directory = data_directory
        self.data_stream = read(self.data_directory)
        self.stats = self.data_stream[0].stats
        self.filename = os.path.basename(self.data_directory)
        print(f"Data loaded from {self.filename} with {len(self.data_stream)} traces.")
    
    def get_plot(self, data = None):
        """
        Plot the data stream.

        :param data: Trace to plot. If None, the original data with no filters will be plotted.

        
        """

        if data is not None:
            tr = data
        else:
            tr = self.data_stream.traces[0].copy()
        
        tr_times = tr.times()
        tr_data = tr.data
        # Initialize figure
        fig,ax = plt.subplots(1,1,figsize=(10,3))

        # Plot trace
        ax.plot(tr_times,tr_data)

        # Make the plot pretty
        ax.set_xlim([min(tr_times),max(tr_times)])
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlabel('Time (s)')
        ax.set_title(f'{self.filename}', fontweight='bold')
        
        plt.show()

    def apply_filter(self, filter_type: str, *args):
        """
        Apply a filter to the data stream.

        :param filter_type: Type of filter to apply. Options are 'bandpass', 'lowpass', 'highpass'
        :param args: Arguments for the filter. For 'bandpass' filter, pass the lower and upper frequency limits. For 'lowpass' and 'highpass' filters, pass the frequency limit.

        :return: Filtered trace
        """
        tr_copy = self.data_stream.traces[0].copy() 
        # tr_copy.data = np.abs(tr_copy.data)

        if filter_type == 'bandpass':
            freqmin, freqmax = args
            tr_copy.filter('bandpass', freqmin=freqmin, freqmax=freqmax)
        elif filter_type == 'lowpass':
            freqmax = args[0]
            tr_copy.filter('lowpass', freq=freqmax)
        elif filter_type == 'highpass':
            freqmin = args[0]
            tr_copy.filter('highpass', freq=freqmin)
        return tr_copy
    

    def LTA_STA_detection_algorithm(self, data = None, sta_len = 120, lta_len = 600, thr_on = 4, thr_off = 1.5):
        if data is not None:
            tr = data
        else:
            tr = self.data_stream.traces[0].copy()

        tr_times = tr.times()
        tr_data = tr.data
        # Sampling frequency of our trace
        df = tr.stats.sampling_rate

        # Run Obspy's STA/LTA to obtain a characteristic function
        # This function basically calculates the ratio of amplitude between the short-term 
        # and long-term windows, moving consecutively in time across the data
        cft = classic_sta_lta(tr_data, int(sta_len * df), int(lta_len * df))

        # Plot characteristic function
        fig,ax = plt.subplots(1,1,figsize=(12,3))
        ax.plot(tr_times,cft)
        ax.set_xlim([min(tr_times),max(tr_times)])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Characteristic function')

        ax.title.set_text('STA/LTA characteristic function')

        plt.show()

        # Play around with the on and off triggers, based on values in the characteristic function
        on_off = np.array(trigger_onset(cft, thr_on, thr_off))
        # The first column contains the indices where the trigger is turned "on". 
        # The second column contains the indices where the trigger is turned "off".

        # Plot on and off triggers
        fig,ax = plt.subplots(1,1,figsize=(12,3))
        for i in np.arange(0,len(on_off)):
            triggers = on_off[i]
            ax.axvline(x = tr_times[triggers[0]], color='red', label='Trig. On')
            ax.axvline(x = tr_times[triggers[1]], color='purple', label='Trig. Off')

        # Plot seismogram
        ax.plot(tr_times,tr_data)
        ax.set_xlim([min(tr_times),max(tr_times)])
        ax.legend()

        plt.show()

    def squared_data(self, data = None):
        if data is not None:
            tr = data
        else:
            tr = self.data_stream.traces[0].copy()

        tr_copy = tr.copy()

        tr_copy.data = np.pow(tr_copy.data, 2)
        return tr_copy
    
if __name__ == '__main__':
    # Path to the data directory
    DATA_DIRECTORY = "./data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1973-07-04HR00_evid00114.mseed"
    # Initialize the DataHandler class
    data_handler = DataHandler(DATA_DIRECTORY)

    # Print the metadata of the data stream
    print(data_handler.stats)

    data_handler.get_plot()
    # data_handler.get_plot(data_handler.apply_filter('bandpass', 0.99999, 1))
    # data_handler.get_plot(data_handler.apply_filter('lowpass', 3))
    # data_handler.get_plot(data_handler.apply_filter('highpass', 2))
    # data_handler.get_plot(data_handler.apply_filter('highpass', 0.9))
    # data_handler.LTA_STA_detection_algorithm(data_handler.apply_filter('highpass', 0.9))

    data_handler.get_plot(data_handler.squared_data())




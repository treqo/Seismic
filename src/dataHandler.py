# Import libraries
import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

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
        self.data_stream.plot()

    def apply_filter(self, filter_type: str, *args):
        """
        Apply a filter to the data stream.

        :param filter_type: Type of filter to apply. Options are 'bandpass', 'lowpass', 'highpass'
        :param args: Arguments for the filter. For 'bandpass' filter, pass the lower and upper frequency limits. For 'lowpass' and 'highpass' filters, pass the frequency limit.

        :return: Filtered trace
        """
        tr_copy = self.data_stream.traces[0].copy() 

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
    
if __name__ == '__main__':
    # Path to the data directory
    DATA_DIRECTORY = "../data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1970-06-26HR00_evid00009.mseed"
    # Initialize the DataHandler class
    data_handler = DataHandler(DATA_DIRECTORY)

    # Print the metadata of the data stream
    print(data_handler.stats)

    data_handler.get_plot()
    data_handler.get_plot(data_handler.apply_filter('bandpass', 0.99999, 1))
    data_handler.get_plot(data_handler.apply_filter('lowpass', 0.1))
    data_handler.get_plot(data_handler.apply_filter('highpass', 0.9))





# Import libraries
import numpy as np
from obspy import read
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import signal
from matplotlib import cm

class DataHandler:
    def __init__(self, data_directory: str, arrival_time: datetime = None):
        """
        Initialize the DataHandler class with the data directory path and optional arrival time.

        :param data_directory: Path to the .mseed file
        :param arrival_time: Optional, absolute time of quake arrival (datetime object)
        """
        self.data_directory = data_directory
        self.data_stream = read(self.data_directory)
        self.stats = self.data_stream[0].stats
        self.arrival_time = arrival_time  # Arrival time of the quake, if known

    def get_trace_data(self):
        """
        Get trace times and data from the seismic data stream.
        :return: tuple (tr_times, tr_data) - times and seismic data
        """
        tr = self.data_stream.traces[0].copy()
        tr_times = tr.times()
        tr_data = tr.data
        return tr_times, tr_data

    def calculate_relative_arrival(self):
        """
        Calculate the relative arrival time of the quake in seconds.
        This method requires self.arrival_time to be set.
        :return: Arrival time in seconds relative to the start of the trace.
        """
        if not self.arrival_time:
            raise ValueError("Arrival time not set. Please provide the arrival time.")
        
        # Get the start time from the seismic trace metadata
        starttime = self.stats.starttime.datetime
        
        # Calculate the relative arrival time (in seconds)
        relative_arrival = (self.arrival_time - starttime).total_seconds()
        return relative_arrival

    def apply_filter(self, filter_type: str, freqmin: float, freqmax: float):
        """
        Apply a filter to the data stream.

        :param filter_type: Type of filter to apply. Options are 'bandpass', 'lowpass', 'highpass'
        :param freqmin: Minimum frequency for the filter
        :param freqmax: Maximum frequency for the filter

        :return: Filtered trace data
        """
        st_filt = self.data_stream.copy() 
        st_filt.filter(filter_type, freqmin=freqmin, freqmax=freqmax)

        # Return filtered trace, times and data
        tr_filt = st_filt.traces[0].copy()
        tr_times_filt = tr_filt.times()
        tr_data_filt = tr_filt.data
        return tr_times_filt, tr_data_filt, tr_filt.stats.sampling_rate

    def plot_time_series_and_spectrogram(self, minfreq=0.01, maxfreq=0.5, plot_title='Filtered Seismic Data'):
        """
        Plot the time series and the corresponding spectrogram.

        :param minfreq: Minimum frequency for the bandpass filter
        :param maxfreq: Maximum frequency for the bandpass filter
        :param plot_title: Title for the time series plot
        """
        # Apply bandpass filter to the data
        tr_times_filt, tr_data_filt, sampling_rate = self.apply_filter('bandpass', minfreq, maxfreq)
        
        # Calculate spectrogram
        f, t, sxx = signal.spectrogram(tr_data_filt, sampling_rate)
        
        # Plot the time series and the spectrogram
        fig = plt.figure(figsize=(10, 10))
        
        # Time series plot
        ax = plt.subplot(2, 1, 1)
        ax.plot(tr_times_filt, tr_data_filt, label="Filtered Seismic Trace")
        
        # Mark quake detection if arrival time is provided
        if self.arrival_time:
            relative_arrival = self.calculate_relative_arrival()
            ax.axvline(x=relative_arrival, color='red', label='Quake Arrival')
            ax.legend(loc='upper left')
        
        # Beautify the time series plot
        ax.set_xlim([min(tr_times_filt), max(tr_times_filt)])
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlabel('Time (s)')
        ax.set_title(plot_title, fontweight='bold')

        # Spectrogram plot
        ax2 = plt.subplot(2, 1, 2)
        vals = ax2.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
        ax2.set_ylabel('Frequency (Hz)', fontweight='bold')
        ax2.set_xlabel('Time (s)', fontweight='bold')
        
        # Mark arrival on the spectrogram plot
        if self.arrival_time:
            ax2.axvline(x=relative_arrival, c='red')
        
        # Add colorbar to spectrogram
        cbar = plt.colorbar(vals, orientation='horizontal')
        cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == '__main__':
    # Path to the data directory
    DATA_DIRECTORY = "../data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1970-06-26HR00_evid00009.mseed"
    
    # Example quake arrival time (replace with actual data)
    ARRIVAL_TIME = datetime(1970, 6, 26, 20, 0, 59, 884000)
    
    # Initialize the DataHandler class with arrival time
    data_handler = DataHandler(DATA_DIRECTORY, arrival_time=ARRIVAL_TIME)

    # Print the metadata of the data stream
    print(data_handler.stats)

    # Plot time series and spectrogram after applying bandpass filter
    data_handler.plot_time_series_and_spectrogram(minfreq=0.01, maxfreq=0.5, plot_title='Filtered Seismic Data with Quake Arrival')

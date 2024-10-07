import numpy as np
import matplotlib.pyplot as plt
from obspy import read
from obspy.signal.trigger import classic_sta_lta, trigger_onset
from datetime import datetime, timedelta

def plot_sta_lta_squared(mseed_file, sta_len=120, lta_len=600, thr_on=4, thr_off=1.5):
    # Read the miniseed file
    st = read(mseed_file)
    tr = st.traces[0].copy()
    
    # Get time and data arrays
    tr_times = tr.times()
    tr_data = tr.data
    
    # Calculate velocity squared
    tr_data_squared = tr_data ** 2
    
    # Calculate sampling frequency
    df = tr.stats.sampling_rate
    
    # Run STA/LTA on squared data
    cft = classic_sta_lta(tr_data_squared, int(sta_len * df), int(lta_len * df))
    
    # Get triggers
    on_off = np.array(trigger_onset(cft, thr_on, thr_off))
    
    # Create the visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot original velocity data
    ax1.plot(tr_times, tr_data)
    ax1.set_title('Original Velocity Data')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity (m/s)')
    
    # Plot squared velocity data
    ax2.plot(tr_times, tr_data_squared)
    ax2.set_title('Squared Velocity Data')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Velocity² (m²/s²)')
    
    # Plot STA/LTA characteristic function
    ax3.plot(tr_times, cft)
    ax3.set_title('STA/LTA Characteristic Function')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('CF Amplitude')
    
    # Add triggers to all plots
    for i in range(len(on_off)):
        triggers = on_off[i]
        trigger_time_on = tr_times[triggers[0]]
        trigger_time_off = tr_times[triggers[1]]
        
        for ax in [ax1, ax2, ax3]:
            ax.axvline(x=trigger_time_on, color='red', label='Trigger On' if i == 0 else "")
            ax.axvline(x=trigger_time_off, color='purple', label='Trigger Off' if i == 0 else "")
    
    # Add legends
    for ax in [ax1, ax2, ax3]:
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    return fig, on_off, tr_times, tr.stats.starttime.datetime

# Example usage:
# mseed_file = './data/lunar/training/data/S12_GradeA/xa.s12.00.mhz.1970-06-26HR00_evid00009.mseed'
# fig, on_off, tr_times, starttime = plot_sta_lta_squared(mseed_file)
# plt.show()

# Function to create detection catalog
def create_detection_catalog(on_off, tr_times, starttime, filename):
    detection_times = []
    filenames = []
    rel_times = []
    
    for triggers in on_off:
        on_time = starttime + timedelta(seconds=tr_times[triggers[0]])
        on_time_str = datetime.strftime(on_time, '%Y-%m-%dT%H:%M:%S.%f')
        
        detection_times.append(on_time_str)
        filenames.append(filename)
        rel_times.append(tr_times[triggers[0]])
    
    return pd.DataFrame({
        'filename': filenames,
        'time_abs(%Y-%m-%dT%H:%M:%S.%f)': detection_times,
        'time_rel(sec)': rel_times
    })
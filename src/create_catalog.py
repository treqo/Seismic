import os
import pandas as pd
import obspy

# Set paths
base_dir = os.path.abspath(os.path.join(os.getcwd(), './data/lunar/test/data'))
output_catalog_dir = os.path.abspath(os.path.join(os.getcwd(), './data/lunar/test/catalogs'))
plots_base_dir = os.path.abspath(os.path.join(os.getcwd(), './data/lunar/test/plots'))

# Ensure output directories exist
os.makedirs(output_catalog_dir, exist_ok=True)
os.makedirs(plots_base_dir, exist_ok=True)

# Function to create a catalog for a given directory
def create_catalog_for_directory(directory):
    catalog = []
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        
        # Check if it's an mseed file
        if file.endswith(".mseed"):
            st = obspy.read(file_path)
            start_time = st[0].stats.starttime
            time_abs = start_time.strftime("%Y-%m-%dT%H:%M:%S.%f")
            # Directly use the result of subtraction, no need for `.seconds`
            time_rel = start_time - obspy.UTCDateTime("1970-01-01T00:00:00.000000")
            evid = f"evid{file.split('_')[-1].split('.')[0]}"
            mq_type = "seismic_mq"
            
            catalog.append({
                "filename": file,
                "time_abs": time_abs,
                "time_rel(sec)": time_rel,
                "evid": evid,
                "mq_type": mq_type
            })

        # Check if it's a CSV file
        elif file.endswith(".csv"):
            time_abs = "N/A"
            time_rel = "N/A"
            evid = f"evid{file.split('_')[-1].split('.')[0]}"
            mq_type = "csv_mq"
            
            catalog.append({
                "filename": file,
                "time_abs": time_abs,
                "time_rel(sec)": time_rel,
                "evid": evid,
                "mq_type": mq_type
            })

    # Create a DataFrame and save it as a CSV catalog
    catalog_df = pd.DataFrame(catalog, columns=["filename", "time_abs", "time_rel(sec)", "evid", "mq_type"])
    catalog_filename = os.path.join(output_catalog_dir, os.path.basename(directory) + "_catalog.csv")
    catalog_df.to_csv(catalog_filename, index=False)

# Iterate over each S* directory
for s_dir in os.listdir(base_dir):
    s_dir_path = os.path.join(base_dir, s_dir)
    
    # Ensure it's a directory
    if os.path.isdir(s_dir_path):
        # Create catalog for the current S* directory
        create_catalog_for_directory(s_dir_path)
        
        # Create a corresponding plots directory for the S* directory
        plots_dir = os.path.join(plots_base_dir, s_dir)
        os.makedirs(plots_dir, exist_ok=True)

print("Catalogs and plot directories have been successfully created.")
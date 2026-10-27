import numpy as np
import matplotlib.pyplot as plt
import pyvisa as visa
import time


def acquire_waveform_data(channels):
    """Acquire waveform data from specified channels on the Rigol DHO924 oscilloscope."""

    # Initialize VISA resource manager
    rm = visa.ResourceManager()
    instruments = rm.list_resources()
    usb = list(filter(lambda x: 'USB' in x, instruments))

    if len(usb) != 1:
        print('Bad instrument list', instruments)
        sys.exit(-1)

    # Open the instrument with increased timeout
    scope = rm.open_resource(usb[0], timeout=10000, chunk_size=1024000)
    scope.write(":STOP")
    time.sleep(0.5)  # Delay to ensure the scope has stopped

    # Dictionary to hold time and voltage data for each channel
    data = {}

    # Loop through each specified channel
    for channel in channels:
        # Set waveform source to the current channel
        scope.write(f":WAV:SOUR CHAN{channel}")

        # Set waveform data format to BYTE for binary data transfer
        scope.write(":WAV:FORM BYTE")

        # Query waveform data in binary format
        try:
            rawdata = scope.query_binary_values(":WAV:DATA?", datatype='B', container=np.array)
            print(f"Channel {channel} data received:", rawdata[:10])  # Print first 10 data points for verification
        except visa.VisaIOError as e:
            print(f"VisaIOError on channel {channel}:", e)
            sys.exit(-1)

        # Retrieve and process timescale and voltage scale for the current channel
        try:
            timescale = float(scope.query(":TIM:SCAL?"))
            timeoffset = float(scope.query(":TIM:OFFS?"))
            voltscale = float(scope.query(f":CHAN{channel}:SCAL?"))
            voltoffset = float(scope.query(f":CHAN{channel}:OFFS?"))
        except visa.VisaIOError as e:
            print(f"Error reading scaling factors for channel {channel}:", e)
            sys.exit(-1)

        # Generate time and voltage data for the current channel
        time_axis = (np.arange(len(rawdata)) - timeoffset) * timescale
        voltage_data = (rawdata - 128) * voltscale + voltoffset

        # Store data in the dictionary
        data[f"CH{channel}"] = {
            "time": time_axis,
            "voltage": voltage_data
        }

    # Close the instrument after data acquisition
    scope.close()
    return data


# Specify the channels to acquire data from (e.g., channels 1 and 2)
channels = [1, 2]
data = acquire_waveform_data(channels)

# Plotting each channel on a separate graph
for channel in channels:
    plt.figure(figsize=(10, 4))
    plt.plot(data[f"CH{channel}"]["time"], data[f"CH{channel}"]["voltage"], label=f"Channel {channel}")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.title(f"Waveform Data from Channel {channel}")
    plt.legend()
    plt.grid(True)
    plt.show()

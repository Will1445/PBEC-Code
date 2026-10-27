import time
import ctypes as ct
import os
import sys
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from picosdk.ps6000 import ps6000 as ps
from picosdk.functions import adc2mV, assert_pico_ok
from ctypes import c_int16, c_int32, byref, c_char_p,c_float
import Components as comp
import socket
import threading

def save_hh_data(data, resolution,tacq,laser_power,stage_position, filter = 0,base_directory='D:/Data'):
    year = datetime.now().strftime('%Y')
    year_month = datetime.now().strftime('%Y%m')
    year_month_day = datetime.now().strftime('%Y%m%d')
    folder_path = os.path.join(base_directory, year, year_month, year_month_day, "g2")
    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.now().strftime('%H%M%S')  # Using HHMMSS format for timestamp
    filename = f"hh_data_{timestamp}_{resolution}_{tacq/1000}_{laser_power:.2f}_{stage_position}_{filter}.txt"
    file_path = os.path.join(folder_path, filename)
    with open(file_path, mode='w') as file:
        file.write("Index\tCounts\n")  # Header row
        for index, count in enumerate(data):
            file.write(f"{index}\t{count}\n")

    return file_path

def save_arrays(channel_data, time_array, stage_position, laser_power, base_directory='D:/Data'):
    year = datetime.now().strftime('%Y')
    year_month = datetime.now().strftime('%Y%m')
    year_month_day = datetime.now().strftime('%Y%m%d')
    folder_path = os.path.join(base_directory, year, year_month, year_month_day, "g1")
    os.makedirs(folder_path, exist_ok=True)
    timestamp = datetime.now().strftime('%H%M%S')  # Using HHMMSS format for timestamp
    for channel, voltage_array in channel_data.items():
        filename_voltage = f"pbec_{timestamp}_{stage_position}_{laser_power}_channel{channel}.txt"
        array_path_voltage = os.path.join(folder_path, filename_voltage)
        np.savetxt(array_path_voltage, voltage_array, fmt='%0.4f')
    filename_time = f"pbec_{timestamp}_{stage_position}_{laser_power}_Time.txt"
    time_path = os.path.join(folder_path, filename_time)
    np.savetxt(time_path, time_array, fmt='%0.4f')

def get_picoscope_data(channels=[0],analogue_offset=0,trig=0,trigsource=0,trigthresh=64,trigdir=2,trigdelay=0,trigms=1000,preTriggerSamples=2500,postTriggerSamples=2500,timebase=8,live_plot=False):
    status = {}
    chandle = c_int16()
    serial = c_char_p()  # Leave serial empty for single device
    status["openunit"] = ps.ps6000OpenUnit(byref(chandle), serial)
    assert status["openunit"] == 0, f"Failed to open PicoScope 6407, error code: {status['openunit']}"
    channel_mapping = {
        0: 'PS6000_CHANNEL_A',
        1: 'PS6000_CHANNEL_B',
        2: 'PS6000_CHANNEL_C',
        3: 'PS6000_CHANNEL_D'
    }
    buffers = {}
    chARange = ps.PS6000_RANGE['PS6000_100MV']  # ±100 mV range, the only range available for PicoScope 6407
    for ch in channels:
        if ch not in channel_mapping:
            raise ValueError("Invalid channel number. Must be 0, 1, 2, or 3.")
        channel = ps.PS6000_CHANNEL[channel_mapping[ch]]
        enabled = 1  # 1 for enabling
        coupling_type = ps.PS6000_COUPLING['PS6000_DC_50R']  # Only DC 50Ω coupling for PicoScope 6407
        bandwidth = ps.PS6000_BANDWIDTH_LIMITER['PS6000_BW_FULL']  # Full bandwidth
        status[f"set_channel_{ch}"] = ps.ps6000SetChannel(
            chandle, channel, enabled, coupling_type, chARange, analogue_offset, bandwidth
        )
        assert status[f"set_channel_{ch}"] == 0, f"Error in setting channel {ch}: {status[f'set_channel_{ch}']}"
        buffers[ch] = {
            'max': (c_int16 * (preTriggerSamples + postTriggerSamples))(),
            'min': (c_int16 * (preTriggerSamples + postTriggerSamples))()  # Used for downsampling
        }
        status[f"setDataBuffers_{ch}"] = ps.ps6000SetDataBuffers(
            chandle, channel, byref(buffers[ch]['max']), byref(buffers[ch]['min']), preTriggerSamples + postTriggerSamples, 0
        )
        assert_pico_ok(status[f"setDataBuffers_{ch}"])
    status["trigger"] = ps.ps6000SetSimpleTrigger(
        chandle, trig, trigsource, trigthresh, trigdir, trigdelay, trigms
    )
    assert_pico_ok(status["trigger"])
    maxSamples = preTriggerSamples + postTriggerSamples
    timeIntervalns = c_float()
    returnedMaxSamples = c_int32()
    status["getTimebase2"] = ps.ps6000GetTimebase2(
        chandle, timebase, maxSamples, byref(timeIntervalns), 1, byref(returnedMaxSamples), 0
    )
    assert_pico_ok(status["getTimebase2"])
    status["runBlock"] = ps.ps6000RunBlock(
        chandle, preTriggerSamples, postTriggerSamples, timebase, 0, None, 0, None, None
    )
    assert_pico_ok(status["runBlock"])
    ready = c_int16(0)
    while ready.value == 0:
        status["isReady"] = ps.ps6000IsReady(chandle, byref(ready))
    overflow = c_int16()
    cmaxSamples = c_int32(maxSamples)
    status["getValues"] = ps.ps6000GetValues(
        chandle, 0, byref(cmaxSamples), 1, 0, 0, byref(overflow)
    )
    assert_pico_ok(status["getValues"])
    maxADC = c_int16(32512)
    channel_data = {}
    for ch in channels:
        adc2mV_buffer = adc2mV(buffers[ch]['max'], chARange, maxADC)
        channel_data[ch] = adc2mV_buffer
    time = np.linspace(0, (cmaxSamples.value - 1) * timeIntervalns.value, cmaxSamples.value) # Create time data
    if live_plot:
        plt.figure(figsize=(5,5))
        for ch in channels:
            plt.plot(time, channel_data[ch][:], label=f"Channel {channel_mapping[ch]}")
        plt.xlabel('Time (ns)')
        plt.ylabel('Voltage (mV)')
        plt.title("PicoScope Data Capture on Multiple Channels")
        plt.legend()
        plt.grid()
        plt.show()
    status["stop"] = ps.ps6000Stop(chandle)
    assert_pico_ok(status["stop"])
    ps.ps6000CloseUnit(chandle) # Close unit (disconnect the scope)
    return channel_data, time # Return channel data and time for further processing

def closeDevices():
    MAXDEVNUM = 8
    if os.name == "nt":
        hhlib = ct.WinDLL("hhlib.dll")
    else:
        hhlib = ct.CDLL("libhh400.so")
    for i in range(0, MAXDEVNUM):
        hhlib.HH_CloseDevice(ct.c_int(i))
    sys.exit(0)

def get_hh_data(binning,offset,tacq,live_plot=False):
    LIB_VERSION   = "3.0"
    MAXDEVNUM     = 8
    MODE_HIST     = 0
    MAXLENCODE    = 6
    HHMAXINPCHAN  = 8
    MAXHISTLEN    = 65536
    FLAG_OVERFLOW = 0x001

    syncDivider        = 1 # You can change this
    syncChannelOffset  = 0 # You can change this (in ps, like a cable delay)
    inputChannelOffset = 0 # You can change this (in ps, like a cable delay)
    syncCFDZeroCross   = 10 # You can change this (in mV)
    syncCFDLevel       = 50 # You can change this (in mV)
    inputCFDZeroCross  = 10 # You can change this (in mV)
    inputCFDLevel      = 50 # You can change this (in mV)

    counts       = [(ct.c_uint * MAXHISTLEN)() for i in range(0, HHMAXINPCHAN)]
    dev          = []
    libVersion   = ct.create_string_buffer(b"", 8)
    hwSerial     = ct.create_string_buffer(b"", 8)
    hwPartno     = ct.create_string_buffer(b"", 8)
    hwVersion    = ct.create_string_buffer(b"", 8)
    hwModel      = ct.create_string_buffer(b"", 16)
    errorString  = ct.create_string_buffer(b"", 40)
    numChannels  = ct.c_int()
    histLen      = ct.c_int()
    resolution   = ct.c_double()
    syncRate     = ct.c_int()
    countRate    = ct.c_int()
    flags        = ct.c_int()
    warnings     = ct.c_int()
    warningstext = ct.create_string_buffer(b"", 16384)

    if os.name == "nt":
        hhlib = ct.WinDLL("hhlib.dll")
    else:
        hhlib = ct.CDLL("libhh400.so")

    def tryfunc(retcode, funcName, measRunning=False):
        if retcode < 0:
            hhlib.HH_GetErrorString(errorString, ct.c_int(retcode))
            print("HH_%s error %d (%s). Aborted." % (funcName, retcode,\
                  errorString.value.decode("utf-8")))
            closeDevices()

    hhlib.HH_GetLibraryVersion(libVersion)

    for i in range(0, MAXDEVNUM):
        retcode = hhlib.HH_OpenDevice(ct.c_int(i), hwSerial)
        if retcode == 0:

            dev.append(i)
        # else:
        #     if retcode == -1: # HH_ERROR_DEVICE_OPEN_FAIL
        #         #print("  %1d        no device" % i)
        #     else:
        #         hhlib.HH_GetErrorString(errorString, ct.c_int(retcode))
        #         print("  %1d        %s" % (i, errorString.value.decode("utf8")))
        #
    if len(dev) < 1:
        print("No device available.")
        closeDevices()

    tryfunc(hhlib.HH_Initialize(ct.c_int(dev[0]), ct.c_int(MODE_HIST), ct.c_int(0)),\
            "Initialize")

    tryfunc(hhlib.HH_GetHardwareInfo(dev[0], hwModel, hwPartno, hwVersion),\
            "GetHardwareInfo")

    tryfunc(hhlib.HH_GetNumOfInputChannels(ct.c_int(dev[0]), byref(numChannels)),\
            "GetNumOfInputChannels")

    tryfunc(hhlib.HH_Calibrate(ct.c_int(dev[0])), "Calibrate")
    tryfunc(hhlib.HH_SetSyncDiv(ct.c_int(dev[0]), ct.c_int(syncDivider)), "SetSyncDiv")

    tryfunc(hhlib.HH_SetSyncCFD(ct.c_int(dev[0]), ct.c_int(syncCFDLevel),\
                                ct.c_int(syncCFDZeroCross)),\
            "SetSyncCFD")

    tryfunc(hhlib.HH_SetSyncChannelOffset(ct.c_int(dev[0]), ct.c_int(syncChannelOffset)),\
            "SetSyncChannelOffset")

    for i in range(0, numChannels.value):
        tryfunc(hhlib.HH_SetInputCFD(ct.c_int(dev[0]), ct.c_int(i), ct.c_int(inputCFDLevel),\
                                     ct.c_int(inputCFDZeroCross)),\
                "SetInputCFD")

        tryfunc(hhlib.HH_SetInputChannelOffset(ct.c_int(dev[0]), ct.c_int(i),\
                                           ct.c_int(inputChannelOffset)),\
            "SetInputChannelOffset")

    tryfunc(hhlib.HH_SetHistoLen(ct.c_int(dev[0]), ct.c_int(MAXLENCODE), byref(histLen)),\
            "SetHistoLen")


    tryfunc(hhlib.HH_SetBinning(ct.c_int(dev[0]), ct.c_int(binning)), "SetBinning")
    tryfunc(hhlib.HH_SetOffset(ct.c_int(dev[0]), ct.c_int(offset)), "SetOffset")
    tryfunc(hhlib.HH_GetResolution(ct.c_int(dev[0]), byref(resolution)), "GetResolution")
    print("Resolution is %1.1lfps" % resolution.value)

    time.sleep(0.4)

    tryfunc(hhlib.HH_GetSyncRate(ct.c_int(dev[0]), byref(syncRate)), "GetSyncRate")
    #print("\nSyncrate=%1d/s" % syncRate.value)

    for i in range(0, numChannels.value):
        tryfunc(hhlib.HH_GetCountRate(ct.c_int(dev[0]), ct.c_int(i), byref(countRate)),\
                "GetCountRate")
        #print("Countrate[%1d]=%1d/s" % (i, countRate.value))

    # New from v1.2: after getting the count rates you can check for warnings
    tryfunc(hhlib.HH_GetWarnings(ct.c_int(dev[0]), byref(warnings)), "GetWarnings")
    if warnings.value != 0:
        hhlib.HH_GetWarningsText(ct.c_int(dev[0]), warningstext, warnings)
        #print("\n\n%s" % warningstext.value.decode("utf-8"))

    tryfunc(hhlib.HH_SetStopOverflow(ct.c_int(dev[0]), ct.c_int(0), ct.c_int(10000)),\
            "SetStopOverflow") # for example only

    tryfunc(hhlib.HH_GetSyncRate(ct.c_int(dev[0]), byref(syncRate)), "GetSyncRate")
    #print("Syncrate=%1d/s" % syncRate.value)

    for i in range(0, numChannels.value):
        tryfunc(hhlib.HH_GetCountRate(ct.c_int(dev[0]), ct.c_int(i), byref(countRate)),\
                "GetCountRate")
        #print("Countrate[%1d]=%1d/s" % (i, countRate.value))

    tryfunc(hhlib.HH_StartMeas(ct.c_int(dev[0]), ct.c_int(tacq)), "StartMeas") #MEAS STARTS HERE !!!!!!!!
    #print("\nMeasuring for %1d milliseconds..." % tacq)

    ctcstatus = ct.c_int(0)
    while ctcstatus.value == 0:
        tryfunc(hhlib.HH_CTCStatus(ct.c_int(dev[0]), byref(ctcstatus)),\
                "CTCStatus")

    tryfunc(hhlib.HH_StopMeas(ct.c_int(dev[0])), "StopMeas")

    for i in range(0, numChannels.value):
        tryfunc(hhlib.HH_GetHistogram(ct.c_int(dev[0]), byref(counts[i]),\
                                  ct.c_int(i), ct.c_int(0)),\
            "GetHistogram")

        integralCount = 0
        for j in range(0, histLen.value):
            integralCount += counts[i][j]

        #print("  Integralcount[%1d]=%1.0lf" % (i,integralCount))

    tryfunc(hhlib.HH_GetFlags(ct.c_int(dev[0]), byref(flags)), "GetFlags")

    if flags.value & FLAG_OVERFLOW > 0:
        print("  Overflow.")

    if live_plot:
        plt.semilogx(counts[0])
        # plt.xlim([500,8000])
        plt.show()
    return(counts[0],resolution.value)


def read_counts_from_file(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    # Find the line after '#counts'
    start_index = next(i for i, line in enumerate(lines) if line.strip() == "#counts") + 1

    # Read and parse the data
    data_lines = lines[start_index:]
    data = [list(map(int, line.strip().split())) for line in data_lines if line.strip()]

    return [row[0] for row in data]  # Extract only the first column (Channel 0)

def plot_counts(counts):
    plt.figure(figsize=(10, 6))
    plt.plot(counts, label='Counts (Channel 0)')
    plt.title('Counts vs. Time Bin (Channel 0)')
    plt.xlabel('Time Bin')
    plt.ylabel('Counts')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def make_data_folder(base_directory='D:/Data'):
    year = datetime.now().strftime('%Y')
    year_month = datetime.now().strftime('%Y%m')
    year_month_day = datetime.now().strftime('%Y%m%d')

    folder_path = os.path.join(
        base_directory,
        year,
        year_month,
        year_month_day
    )

    os.makedirs(folder_path, exist_ok=True)
    return folder_path


def format_for_filename(x):
    """Keep decimal points in filenames but remove spaces."""
    return str(x).strip().replace(" ", "")


def power_meter_worker(
        powermeter,
        stop_event,
        measured_power,
        measured_time,
        dt=0.1
):
    t0 = time.time()

    while not stop_event.is_set():
        try:
            reading_uW = 1e6 * powermeter.take_power_reading()
            measured_power.append(reading_uW)
            measured_time.append(time.time() - t0)

        except Exception:
            pass

        time.sleep(dt)

from mdt69x import Controller
controller = Controller("COM4")

def move_piezo(voltage):
    controller.set_x_voltage(voltage)
    controller.get_x_voltage()
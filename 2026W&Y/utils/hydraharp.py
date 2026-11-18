"""Small HHLIB v3 histogram adapter, following the lab's existing API calls.

Only the device opened by this object is closed. Every return code is checked.
No measurement starts on construction. The explicit settings are saved by callers.
"""
import ctypes as ct
import time


class HydraHarp:
    def __init__(self):
        self.lib=ct.WinDLL(r'D:\Control\HydraHarp\hhlib.dll')
        self.device=None
        self.running=False
        defs={
            'GetLibraryVersion':[ct.c_void_p], 'GetErrorString':[ct.c_void_p,ct.c_int],
            'OpenDevice':[ct.c_int,ct.c_void_p], 'CloseDevice':[ct.c_int],
            'Initialize':[ct.c_int,ct.c_int,ct.c_int],
            'GetHardwareInfo':[ct.c_int,ct.c_void_p,ct.c_void_p,ct.c_void_p],
            'GetNumOfInputChannels':[ct.c_int,ct.POINTER(ct.c_int)],
            'Calibrate':[ct.c_int], 'SetSyncDiv':[ct.c_int,ct.c_int],
            'SetSyncCFD':[ct.c_int,ct.c_int,ct.c_int],
            'SetSyncChannelOffset':[ct.c_int,ct.c_int],
            'SetInputCFD':[ct.c_int,ct.c_int,ct.c_int,ct.c_int],
            'SetInputChannelOffset':[ct.c_int,ct.c_int,ct.c_int],
            'SetHistoLen':[ct.c_int,ct.c_int,ct.POINTER(ct.c_int)],
            'SetBinning':[ct.c_int,ct.c_int], 'SetOffset':[ct.c_int,ct.c_int],
            'GetResolution':[ct.c_int,ct.POINTER(ct.c_double)],
            'SetStopOverflow':[ct.c_int,ct.c_int,ct.c_uint],
            'GetSyncRate':[ct.c_int,ct.POINTER(ct.c_int)],
            'GetCountRate':[ct.c_int,ct.c_int,ct.POINTER(ct.c_int)],
            'GetFlags':[ct.c_int,ct.POINTER(ct.c_int)],
            'GetWarnings':[ct.c_int,ct.POINTER(ct.c_int)],
            'GetWarningsText':[ct.c_int,ct.c_void_p,ct.c_int],
            'ClearHistMem':[ct.c_int], 'StartMeas':[ct.c_int,ct.c_int],
            'CTCStatus':[ct.c_int,ct.POINTER(ct.c_int)], 'StopMeas':[ct.c_int],
            'GetHistogram':[ct.c_int,ct.POINTER(ct.c_uint),ct.c_int,ct.c_int],
            'GetElapsedMeasTime':[ct.c_int,ct.POINTER(ct.c_double)]}
        for name,args in defs.items():
            f=getattr(self.lib,'HH_'+name);f.restype=ct.c_int;f.argtypes=args
        self.settings={'mode':'histogram','clock':'internal','binning':7,'histogram_offset_ns':0,
            'sync_divider':1,'sync_CFD_level_mV':50,'sync_CFD_zero_cross_mV':10,'sync_offset_ps':0,
            'input_CFD_level_mV':50,'input_CFD_zero_cross_mV':10,'input_offset_ps':0,
            'settings_source':'Existing Measurement_functions.py; binning chosen to retain observed GUI 128 ps resolution'}
        try:
            version=ct.create_string_buffer(8);serial=ct.create_string_buffer(8)
            self.call('GetLibraryVersion',version)
            self.call('OpenDevice',0,serial);self.device=0
            self.call('Initialize',0,0,0)
            buffers=[ct.create_string_buffer(n) for n in (16,8,8)]
            self.call('GetHardwareInfo',0,*buffers)
            channels=ct.c_int();self.call('GetNumOfInputChannels',0,ct.byref(channels))
            self.channels=channels.value
            if not 1<=self.channels<=8: raise RuntimeError('Unexpected channel count')
            self.call('Calibrate',0)
            self.call('SetSyncDiv',0,1);self.call('SetSyncCFD',0,50,10)
            self.call('SetSyncChannelOffset',0,0)
            for ch in range(self.channels):
                self.call('SetInputCFD',0,ch,50,10)
                self.call('SetInputChannelOffset',0,ch,0)
            n=ct.c_int();self.call('SetHistoLen',0,6,ct.byref(n));self.n=n.value
            self.call('SetBinning',0,7);self.call('SetOffset',0,0)
            res=ct.c_double();self.call('GetResolution',0,ct.byref(res))
            self.call('SetStopOverflow',0,1,4294967295)
            self.settings.update(library_version=version.value.decode(),serial=serial.value.decode(),
                hardware=[b.value.decode() for b in buffers],input_channels=self.channels,
                histogram_bins=self.n,resolution_ps=res.value,stop_on_overflow=True)
            time.sleep(0.4)
        except BaseException:
            self.close();raise

    def call(self,name,*args):
        rc=getattr(self.lib,'HH_'+name)(*args)
        if rc<0:
            error=ct.create_string_buffer(80);self.lib.HH_GetErrorString(error,rc)
            raise RuntimeError('HH_%s: %d %s'%(name,rc,error.value.decode(errors='replace')))
        return rc

    def rates(self):
        sync=ct.c_int();self.call('GetSyncRate',self.device,ct.byref(sync))
        inputs=[]
        for ch in range(self.channels):
            rate=ct.c_int();self.call('GetCountRate',self.device,ch,ct.byref(rate));inputs.append(rate.value)
        return {'sync_cps':sync.value,'input_cps':inputs}

    def status(self):
        flags=ct.c_int();warnings=ct.c_int()
        self.call('GetFlags',self.device,ct.byref(flags));self.call('GetWarnings',self.device,ct.byref(warnings))
        buf=ct.create_string_buffer(16384)
        if warnings.value:self.call('GetWarningsText',self.device,buf,warnings.value)
        return {'flags':flags.value,'warnings':warnings.value,'warning_text':buf.value.decode(errors='replace')}

    def start(self,seconds):
        if not 0<seconds<=60:raise ValueError('Pilot blocks limited to 60 seconds')
        self.call('ClearHistMem',self.device)
        self.call('StartMeas',self.device,round(seconds*1000));self.running=True

    def complete(self):
        done=ct.c_int();self.call('CTCStatus',self.device,ct.byref(done));return bool(done.value)

    def stop_read(self):
        self.call('StopMeas',self.device);self.running=False
        elapsed=ct.c_double();self.call('GetElapsedMeasTime',self.device,ct.byref(elapsed))
        histograms=[]
        for ch in range(self.channels):
            counts=(ct.c_uint*self.n)();self.call('GetHistogram',self.device,counts,ch,0)
            histograms.append(list(counts))
        return {'elapsed_ms':elapsed.value,'histograms':histograms,'status':self.status()}

    def close(self):
        if self.device is not None:
            try:
                if self.running:self.call('StopMeas',self.device)
            finally:
                self.call('CloseDevice',self.device);self.device=None;self.running=False


if __name__=='__main__':
    import json
    hh=HydraHarp()
    try:
        print(json.dumps({'settings':hh.settings,'rates':hh.rates(),'status':hh.status()},indent=2))
    finally:hh.close()

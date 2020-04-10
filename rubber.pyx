# distutils: language = c++

from cython cimport view
import numpy as np
cimport numpy as np
from libcpp cimport bool
from libc.stdlib cimport malloc, free
import math
import time

from RubberBandStretcher cimport RubberBandStretcher

cdef class Rubber(object):
    cdef RubberBandStretcher* stretcher  # hold a pointer to the C++ instance which we're wrapping
    cdef bool realtime

    # OPTIONS
    OptionProcessOffline       = 0x00000000
    OptionProcessRealTime      = 0x00000001

    OptionStretchElastic       = 0x00000000
    OptionStretchPrecise       = 0x00000010

    OptionTransientsCrisp      = 0x00000000
    OptionTransientsMixed      = 0x00000100
    OptionTransientsSmooth     = 0x00000200

    OptionDetectorCompound     = 0x00000000
    OptionDetectorPercussive   = 0x00000400
    OptionDetectorSoft         = 0x00000800

    OptionPhaseLaminar         = 0x00000000
    OptionPhaseIndependent     = 0x00002000

    OptionThreadingAuto        = 0x00000000
    OptionThreadingNever       = 0x00010000
    OptionThreadingAlways      = 0x00020000

    OptionWindowStandard       = 0x00000000
    OptionWindowShort          = 0x00100000
    OptionWindowLong           = 0x00200000

    OptionSmoothingOff         = 0x00000000
    OptionSmoothingOn          = 0x00800000

    OptionFormantShifted       = 0x00000000
    OptionFormantPreserved     = 0x01000000

    OptionPitchHighSpeed       = 0x00000000
    OptionPitchHighQuality     = 0x02000000
    OptionPitchHighConsistency = 0x04000000

    OptionChannelsApart        = 0x00000000
    OptionChannelsTogether     = 0x10000000

    # Preset Options
    DefaultOptions             = 0x00000000
    PercussiveOptions          = 0x00102000


    def __cinit__(self, int sample_rate=44100,
                        int channels=1,
                        int options=Rubber.DefaultOptions,
                        bool realtime=False,
                        double timeRatio=1.0,
                        double pitchScale=1.0):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive!")
        if channels <= 0:
            raise ValueError("channels must be positive!")
        
        self.realtime = realtime
        if self.realtime:
            options = options | Rubber.OptionProcessRealTime
        self.stretcher = new RubberBandStretcher(sample_rate, channels, options, timeRatio, pitchScale)
        print(f"Options: {hex(options)}")

    def __dealloc__(self):
        del self.stretcher
    
    def reset(self):
        self.stretcher.reset()

    # Getters and Setters
    def get_time_ratio(self) -> double:
        return self.stretcher.getTimeRatio()
    
    def set_time_ratio(self, double ratio):
        """NOTE: has no effect in offline mode after the first call to study() or process() """
        self.stretcher.setTimeRatio(ratio)
    
    def get_pitch_scale(self) -> double:
        return self.stretcher.getPitchScale()
    
    def set_pitch_scale(self, double scale):
        """NOTE: has no effect in offline mode after the first call to study() or process() """
        self.stretcher.setPitchScale(scale)
    
    def get_latency(self) -> size_t:
        return self.stretcher.getLatency()
    
    def get_channel_count(self) -> size_t:
        return self.stretcher.getChannelCount()

    # Option Setters
    def set_transient_option(self, int options):
        self.stretcher.setTransientsOption(options)
    
    def set_detector_option(self, int options):
        self.stretcher.setDetectorOption(options)
    
    def set_phase_option(self, int options):
        self.stretcher.setPhaseOption(options)
    
    def set_formant_option(self, int options):
        self.stretcher.setFormantOption(options)
    
    def set_pitch_option(self, int options):
        self.stretcher.setPitchOption(options)

    # processing
    def set_expected_input_duration(self, size_t samples):
        if samples <= 0:
            raise ValueError("samples must be greater than zero!")
        self.stretcher.setExpectedInputDuration(samples)
    
    def set_max_process_size(self, size_t samples):
        if samples <= 0:
            raise ValueError("samples must be greater than zero!")
        self.stretcher.setMaxProcessSize(samples)
    
    def available(self) -> int:
        return self.stretcher.available()
    
    def get_samples_required(self) -> size_t:
        return self.stretcher.getSamplesRequired()
    
    cdef void _fill_double_ptr(self, float **ptr, float[:, ::1] data, size_t sample_offset=0):
        cdef size_t channels = data.shape[0]

        if not ptr: 
            raise MemoryError
        try:
            for i in range(channels): 
                ptr[i] = &data[i, sample_offset]
        except Exception as e:
            print(e)
    
    cdef void _study_ptr(self, float** input_ptr, size_t samples, bool final): 
        self.stretcher.study(input_ptr, samples, final)

    def study(self, np.ndarray input, bool final, sample_offset=0):
        # safety checks
        if sample_offset < 0:
            sample_offset = 0

        # create the memoryview
        cdef float[:, ::1] input_memview = self._preprocess_numpy_audio(input)

        # these are nice to have
        cdef size_t channels = input_memview.shape[0]
        cdef size_t samples = input_memview.shape[1]

        # allocate memory for double ptr
        cdef float ** input_ptr = <float **>malloc(channels * sizeof(float*))
        if not input_ptr: 
            raise MemoryError
        
        self._fill_double_ptr(input_ptr, input_memview)

        # study() and then free the ptr
        self.stretcher.study(input_ptr, samples, final)
        free(input_ptr)
    
    cdef void _process_ptr(self, float** input_ptr, size_t samples, bool final): 
        self.stretcher.process(input_ptr, samples, final)
    
    def process(self, np.ndarray input, bool final, sample_offset=0):
        # safety checks
        if sample_offset < 0:
            sample_offset = 0

        # create the memoryview
        cdef float[:, ::1] input_memview = self._preprocess_numpy_audio(input)

        # nice to have
        cdef size_t channels = input_memview.shape[0]
        cdef size_t samples = input_memview.shape[1]

        # allocate double ptr
        cdef float ** input_ptr = <float **>malloc(channels * sizeof(float*))
        if not input_ptr: 
            raise MemoryError

        self._fill_double_ptr(input_ptr, input_memview)

        # process and then free the double pointer
        self.stretcher.process(input_ptr, samples, final)
        free(input_ptr)

    cdef void _retrieve_ptr(self, float **output_ptr, size_t samples):
        self.stretcher.retrieve(output_ptr, samples)

    def retrieve(self, int samples=0):
        # get dimensions of the data we'll receive
        cdef size_t channels = self.stretcher.getChannelCount()

        # get number of samples and return empty array if no samples available
        if samples <= 0:
            samples = self.stretcher.available()
            if samples <= 0:
                return np.zeros((0, channels))
        
        # create output memview
        output = np.zeros((channels, samples), dtype=np.float32)
        cdef float[:, ::1] output_memview = output

        # create double pointer for output
        cdef float ** output_ptr = <float **>malloc(channels * sizeof(float*))
        if not output_ptr: 
            raise MemoryError
        self._fill_double_ptr(output_ptr, output_memview)

        # retrieve data
        self.stretcher.retrieve(output_ptr, samples)

        free(output_ptr)
    
        return output.T

    def _preprocess_numpy_audio(self, np.ndarray arr):
        if arr.dtype != np.float32:
            raise ValueError("audio data should be of dtype: float32")

        num_channels = self.get_channel_count()
        if num_channels > 1:
            # enforce 2d array
            if arr.ndim != 2:
                raise ValueError("audio data should have 2 dimensions")
            if arr.shape[1] != num_channels:
                raise ValueError(f"audio data should have {num_channels} channels")

            arr = arr.T  # get channels as first dim
        else:
            # just 1 channel, allow 1d array
            if arr.ndim == 1:
                arr = np.expand_dims(arr, axis=0)  # insert single channel as first dim
            elif arr.ndim == 2:
                if arr.shape[1] != num_channels:
                    raise ValueError(f"audio data should have {num_channels} channels")
                
                arr = arr.T  # get channels as first dim
            else:
                raise ValueError("audio data should not have more than 2 dimensions")
        
        # finally make contiguous
        if not arr.flags['C_CONTIGUOUS']:
            print("making audio data c contiguous")
            arr = np.ascontiguousarray(arr)
        
        return arr


    def stretch(self, data: np.ndarray, time_scale: float, unsigned int block_size=1024, bool final=False):
        #self.stretcher.reset()
        # get data size
        cdef size_t samples = data.shape[0]
        cdef size_t channels = data.shape[1]
        # update time scale
        self.set_time_ratio(time_scale)

        # get the input as double pointer
        cdef float[:, ::1] input_memview = self._preprocess_numpy_audio(data)
        cdef float ** input_ptr = <float **>malloc(channels * sizeof(float*))
        if not input_ptr: 
            raise MemoryError
        
        if not self.realtime:
            print("Studying...")
            self._fill_double_ptr(input_ptr, input_memview)
            self._study_ptr(input_ptr, samples, True)
            print("Finished Studying!")


        # preallocate output
        output_length = math.ceil((samples * time_scale) / block_size) * block_size
        output = np.zeros((channels, output_length), dtype=np.float32)
        
        # get output to double pointer
        cdef float[:, ::1] output_memview = output
        cdef float ** output_ptr = <float **>malloc(channels * sizeof(float*))
        if not output_ptr: 
            raise MemoryError
        
        cdef size_t in_idx = 0
        cdef size_t out_idx = 0
        cdef bool is_final = False
        cdef size_t next_idx = 0
        cdef int avail = 0
        while in_idx < samples:
            next_idx = in_idx + block_size
            is_final = False
            if next_idx >= samples:
                next_idx = samples
                is_final = final if self.realtime else True  # if offline, last block is final, otherwise use the param
            
            self._fill_double_ptr(input_ptr, input_memview, sample_offset=in_idx)
            self._process_ptr(input_ptr, next_idx - in_idx, is_final)

            avail = self.stretcher.available()
            if avail:
                self._fill_double_ptr(output_ptr, output_memview, sample_offset=out_idx)
                self._retrieve_ptr(output_ptr, avail)
                print(f"retrieved: {avail}")
                out_idx += avail
    
            print(f"Processing {100*in_idx/samples}%")
            in_idx = next_idx
        
        # what value to compare to in our final while loop
        cdef int min_avail = 0 if not self.realtime or final else 1
        time.sleep(0.1)  # TODO: get rid of these!

        avail = self.stretcher.available()
        while avail >= min_avail:
            print(f"Retreiving {avail} at idx {out_idx}")
            self._fill_double_ptr(output_ptr, output_memview, sample_offset=out_idx)
            self._retrieve_ptr(output_ptr, avail)
            out_idx += avail
            avail = self.stretcher.available()
            time.sleep(0.1)  # TODO: get rid of these!

        
        free(input_ptr)
        free(output_ptr)
        
        return output[:,:out_idx].T

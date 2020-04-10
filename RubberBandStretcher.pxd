from libc.stddef cimport size_t
from libcpp cimport bool

cdef extern from "RubberBandStretcher.h" namespace "RubberBand":

    cdef cppclass RubberBandStretcher:
        ctypedef int Options
        
        RubberBandStretcher(size_t sampleRate,
                        size_t channels,
                        Options options,
                        double initialTimeRatio,
                        double initialPitchScale) except + 
        
        void reset()
        void setTimeRatio(double ratio)
        void setPitchScale(double scale)
        double getTimeRatio() const
        double getPitchScale() const
        size_t getLatency() const
        size_t getChannelCount() const
        void setTransientsOption(Options options)
        void setDetectorOption(Options options)
        void setPhaseOption(Options options)
        void setFormantOption(Options options)
        void setPitchOption(Options options)
        void setExpectedInputDuration(size_t samples)
        void setMaxProcessSize(size_t samples)

        size_t getSamplesRequired() const;
        int available() const

        void study(const float *const *input, size_t samples, bool final)
        void process(const float *const *input, size_t samples, bool final)

        size_t retrieve(float *const *output, size_t samples) const

"""
void setKeyFrameMap(const std::map<size_t, size_t> &)


void calculateStretch()
void setDebugLevel(int level)
static void setDefaultDebugLevel(int level)
"""
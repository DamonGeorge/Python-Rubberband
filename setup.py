from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("rubberband", ["audio_stretcher.pyx"],
              language="c++",
              include_dirs=["rubberband/", np.get_include()],
              libraries=["fftw3", "fftw3f", "samplerate", "sndfile"],
              library_dirs=["/usr/local/lib"],
              extra_objects=["lib/mac_osx/librubberband.a"]
              )
]

setup(
    name='Rubberband Python Wrapper',
    ext_modules=cythonize(extensions, annotate=True, compiler_directives={"language_level": 3}),
    zip_safe=False,
)

# setup(ext_modules=cythonize(extensions))

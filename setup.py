from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension("rubber", ["rubber.pyx"],
              language="c++",
              include_dirs=["rubberband/rubberband", np.get_include()],
              libraries=["fftw3", "fftw3f", "samplerate", "sndfile"],
              library_dirs=["/usr/local/lib"],
              extra_objects=["rubberband/lib/librubberband.a"]
              )
]

setup(
    name='Rubberband Python Wrapper',
    ext_modules=cythonize(extensions, annotate=True, compiler_directives={"language_level": 3}),
    zip_safe=False,
)

# setup(ext_modules=cythonize(extensions))

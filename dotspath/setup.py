from setuptools import setup, Extension

dotspath_module = Extension('dotspath', sources = ['module.cpp'])

setup(
    name='dotspath',
    version='1.0',
    description='Dots game path utils',
    ext_modules=[dotspath_module]
)

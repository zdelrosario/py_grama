from setuptools import setup, find_packages

setup(
    name="py_grama",
    version="0.1dev",
    packages=['grama', 'grama.fit', 'grama.data', 'grama.models'],
    package_data={'grama.data': ['*.csv']},
    license="MIT",
    long_description=open('README.md').read()
)

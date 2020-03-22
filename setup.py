from setuptools import setup, find_packages

setup(
    name="py_grama",
    author="Zachary del Rosario",
    author_email="zdelrosario@outlook.com",
    version="0.1.2",
    packages=['grama', 'grama.fit', 'grama.data', 'grama.models'],
    package_data={'grama.data': ['*.csv']},
    license="MIT",
    url="https://github.com/zdelrosario/py_grama",
    description="A grammar of model analysis",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)

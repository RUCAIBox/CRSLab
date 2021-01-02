from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

try:
    import torch
    import torch_geometric
except:
    raise Exception('Please install PyTorch and PyTorch Geometric manually first.\n' + 
                    'View CRSLab GitHub page for more information: https://github.com/RUCAIBox/CRSLab')
    exit(1)

install_requires = [
    'numpy~=1.19.4',
    'sentencepiece<0.1.92',
    'dataclasses~=0.7',
    'transformers~=4.1.1',
    'fasttext~=0.9.2',
    'pkuseg~=0.0.25',
    'pyyaml~=5.3.1',
    'tqdm~=4.55.0',
    'loguru~=0.5.3',
    'nltk~=3.4.4',
    'requests~=2.25.1',
    'scikit-learn~=0.24.0',
    'fuzzywuzzy~=0.18.0',
]

setup_requires = []

classifiers = ["License :: OSI Approved :: MIT License"]

long_description = 'CRSLab is an open-source toolkit developed based ' \
                   'on Python and Pytorch for reproducing and developing ' \
                   'conversational recommender systems. View CRSLab GitHub ' \
                   'page for more information: https://github.com/RUCAIBox/CRSLab'

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    install_requires.extend(setup_requires)

setup(
    name='crslab',
    version=
    '0.1.1',  # please remember to edit crslab/__init__.py in response, once updating the version
    description='An Open-Source Toolkit for Building Conversational Recommender System',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/RUCAIBox/CRSLab',
    author='CRSLabTeam',
    author_email='francis_kun_zhou@163.com',
    packages=[
        package for package in find_packages()
        if package.startswith('crslab')
    ],
    include_package_data=True,
    install_requires=install_requires,
    setup_requires=setup_requires,
    zip_safe=False,
    classifiers=classifiers,
)
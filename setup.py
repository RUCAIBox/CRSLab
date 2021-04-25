from setuptools import setup, find_packages

try:
    import torch
    import torch_geometric
except Exception:
    raise Exception('Please install PyTorch and PyTorch Geometric manually first.\n' +
                    'View CRSLab GitHub page for more information: https://github.com/RUCAIBox/CRSLab')

setup_requires = []

install_requires = [
    'numpy~=1.19.4',
    'sentencepiece<0.1.92',
    "dataclasses~=0.7;python_version<'3.7'",
    'transformers~=4.1.1',
    'fasttext~=0.9.2',
    'pkuseg~=0.0.25',
    'pyyaml~=5.4',
    'tqdm~=4.55.0',
    'loguru~=0.5.3',
    'nltk~=3.4.4',
    'requests~=2.25.1',
    'scikit-learn~=0.24.0',
    'fuzzywuzzy~=0.18.0',
    'tensorboard~=2.4.1',
]

classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Scientific/Engineering :: Human Machine Interfaces"
]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='crslab',
    version='0.1.1',  # please remember to edit crslab/__init__.py in response, once updating the version
    author='CRSLabTeam',
    author_email='francis_kun_zhou@163.com',
    description='An Open-Source Toolkit for Building Conversational Recommender System(CRS)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/RUCAIBox/CRSLab',
    packages=[
        package for package in find_packages()
        if package.startswith('crslab')
    ],
    classifiers=classifiers,
    install_requires=install_requires,
    setup_requires=setup_requires,
    python_requires='>=3.6',
)

#!/bin/bash
python --version

pip install h5py
pip install typing-extensions
pip install wheel

pip install --upgrade cython
pip install --upgrade azure-cli
pip install --upgrade azureml-sdk
pip install -r requirements.txt
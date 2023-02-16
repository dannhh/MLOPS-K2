#!/bin/bash
python --version

pip install h5py
pip install typing-extensions
pip install wheel

pip install --upgrade cython

# The Azure command-line interface (Azure CLI) is a set of commands used to 
# create and manage Azure resources. 
pip install --upgrade azure-cli

# The Azure Machine Learning SDK is used to build and run machine learning 
# workflows upon the Azure Machine Learning service, interact with the service 
# in any Python environment, including Jupyter Notebooks or Python IDE.
pip install --upgrade azureml-sdk

pip install azure.keyvault.secrets
pip install azureml.pipeline

pip install -r requirements.txt

pip install --upgrade numpy
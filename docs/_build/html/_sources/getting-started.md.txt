# Getting Started

This setup was tested on linux and m1 mac

## Setup 
1. The raw data files [from kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat?resource=download&select=mitbih_train.csv) must be placed in the `data/raw` 

Minimally requires `mitbih_train.csv` and `mitbih_test.csv` to be there
```
├── data
        ├── processed    
        └── raw
            └── mitbih_train.csv
            └── mitbih_test.csv
```

2. A conda must be installed on the system. 
- Alternatively, you can use the .devcontainer or dockerfile to create a docker environment with conda installed.

## Creating the environment 
Create and enter the environment 
```
conda create -n telfer-ecg python=3.11
activate activate telfer-ecg
```

## Installing Python Dependencies
```
make requirements
```

## Building Dataset

To build the `data/processed` datafiles, simply run
```
make data
```
this will process the data with augmentation and feature extraction.

## Running the Hyperparameter Optimization

To run hyperparameter optimization, simply run
```
make run
```

## Createing the Visualizations

The visualizations are not part of the pipelines, they can be created by running the jupyter notebooks in `/notebooks` folder

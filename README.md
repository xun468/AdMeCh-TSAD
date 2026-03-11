# Addressing Methodological Challenges in Time Series Anomaly Detection
Official implementation of Addressing Methodological Challenges in Time Series Anomaly Detection

## Installation 
Recommended requirements to install and run: 
- `git`
- `conda`

### Clone repo: 
```bash
git clone [https://github.com/TheDatumOrg/TSB-AD.git](https://github.com/xun468/AdMeCh-TSAD.git)
```

### Create Environment: 
```bash
conda create -n your_env python=3.10.4   
conda activate your_env
```

### Install dependencies in requirements.txt:
```bash
pip install -r requirements.txt
```
Note: This does not include CUDA, which may be installed using: 
```bash
conda install pytorch==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
```
### Additional dependencies: 
- `jupyter` (either notebook or lab) 

## Datasets 
Datasets used in the project can be acquired from the following sources: 

SWaT : https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/ On request from iTrust labs. We use `SWaT_Dataset_Attack_v0.csv` and `SWaT_Dataset_Normal_v1.csv` from `SWaT.A1 & A2_Dec 2015`

WaDI : https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/ On request from iTrust labs. We use `WADI_14days.csv` and `WADI_attackdata.csv` from `WaDi.A1_9 Oct 2017` 

SMAP/MSL : https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl

SMD : https://www.kaggle.com/datasets/mgusat/smd-onmiad

UCR : https://wu.renjie.im/research/anomaly-benchmarks-are-flawed/#ucr-time-series-anomaly-archive

The expected filestructure can be seen in `datasets\`. Dataloading is handled in `utils\preprocess.py`

## Usage 
Examples of how experiments and evaluations are run can be found in `AdMeChTSAD.ipynb`. 

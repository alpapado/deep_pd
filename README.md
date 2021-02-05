# Unobtrusive detection of Parkinson’s disease from multi-modal and in-the-wild sensor data using deep learning techniques
Supplementary code and pre-trained models for the [paper](https://www.nature.com/articles/s41598-020-78418-8) published in Scientific Reports.

Dataset can be found [here](https://zenodo.org/record/4311175)

## Instructions
Preparatory steps
```bash
git clone https://github.com/alpapado/deep_pd
cd deep_pd
conda env create -f environment.yml
wget https://zenodo.org/record/4311175/files/data.zip
unzip data.zip -d data
```
Command to reproduce SData experiment (using pre-trained models)
```bash
python deep_pd_mil.py with seed=42 train_params.evaluation_on=sdata imu_params.checkpoint=True typing_params.checkpoint=True
```

Command to reproduce GData experiment (using pre-trained models)
```bash
python deep_pd_mil.py with seed=42 train_params.evaluation_on=gdata imu_params.checkpoint=True typing_params.checkpoint=True
```

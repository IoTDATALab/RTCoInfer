# RTCoInfer: Real-time Edge-Cloud Collaborative CNN Inference for Stream Analytics on Ubiquitous Images.

The implementation of paper : RTCoInfer: Real-time Edge-Cloud Collaborative CNN Inference for Stream Analytics on Ubiquitous Images. 

We have conducted comparative experiments with some of the best engineering in the field.

|RTCoInfer|CLIO|SPINN|MobileViT
|:-:|:-:|:-:|:-:|
|Real-time Edge-Cloud Collaborative CNN Inference for Stream Analytics on Ubiquitous Images. For more details, please click [RTCoInfer](https://github.com/IoTDATALab/RTCoInfer)|Enabling automatic compilation of deep learning pipelines across IoT and Cloud. For more details, please click [CLIO](https://github.com/IoTDATALab/RTCoInfer/tree/main/CLIO) |Synergistic progressive inference of neural networks over device and cloud. For more details, please click [SPINN](https://github.com/IoTDATALab/RTCoInfer/tree/main/SPINN) |Light-weight, General-purpose, and Mobile-friendly Vision Transformer. For more details, please click [MobileViT](https://github.com/IoTDATALab/RTCoInfer/tree/main/MobileViT)


## Install
Clone repo and install requirements.txt in a Python>=3.7.0 environment, including PyTorch>=1.3.1.
```
git clone https://github.com/IoTDATALab/RTCoInfer.git  # clone
cd RTCoInfer
pip install -r requirements.txt  # install
```
## Data Resource
* Experiments are running on the dataset StanfordCars
* You may follow the data preparation guide [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).

## Running the experiments
* modify the configuration files in apps
```
modify the item "dataset_dir:" to your dataset 
```
* Train the model without compression and get a initialization_model.pt
```
python train_only_one.py app:apps/train_only_one.yml
```
* Get the clusters of compression setups by Clustering method.
* Here exists an interactive interface after one epoch training. User can input the cluster number and eps for DBSCAN, and we have a guide for user to achieve the defined cluster number. 
```
python cluster.py app:apps/cluster.yml
```
* Train the SWitchable CNN model.
```
python swcnn_train.py app:apps/sw_mobilenet_v2.yml
```
* Test SWitchable CNN model
```
add the configuration "test_only: True" in apps/sw_mobilenet_v2.yml
add the configuration "test_model: ./logs/best_model.pt" in apps/sw_mobilenet_v2.yml
python swcnn_train.py app:apps/sw_mobilenet_v2.yml
```

## Ablation experiments
The training effect using initialization, clustering, and distillation techniques is significantly improved.
<img src = assets/img/results.png width=60% />  

The difference between the **top1 error** data generated by training different switchable training methods and the data generated by training individually.

<div class="center">

|Method|max|min|mean|
|:---------:|:-----:|:-----:|:------:|
|noI-D-C|0.552|0.406|0.4382|
|noI-noD-noC|0.503|0.348|0.3716|
|I-noD-C|0.181|0.013|0.0347|
|noI-noD-C|0.14|0.01|0.0329|
|I-noD-noC|0.153|0.007|0.0265|
|I-D-noC|0.095|**0.0**|0.0118|
|**I-D-C**|**0.013**|**0.0**|**0.0027**|

</div>

## Performance verification with switchable training weights
The performance of the switchable training weights is similar to the performance of the separately trained weights.

You can download the individually trained weights by clicking on [this link](https://drive.google.com/drive/folders/1RGwLZUXjmtA3b_ok1v3Lcrwx-dGeidMU?usp=sharing).

After putting them under the current folder, you can test the performance of the downloaded weights with the following command.
```
python exp_test.py
```

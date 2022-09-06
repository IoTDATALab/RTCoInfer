# RTCoInfer: Real-time Edge-Cloud Collaborative CNN Inference for Stream Analytics on Ubiquitous Images.

The implementation of paper : RTCoInfer: Real-time Edge-Cloud Collaborative CNN Inference for Stream Analytics on Ubiquitous Images. 

## Install
Clone repo and install requirements.txt in a Python>=3.7.0 environment, including PyTorch>=1.3.1.
```
git clone https://github.com/fairyw98/RTCoInfer.git  # clone
cd RTCoInfer
pip install -r requirements.txt  # install
```
## Data Resource
* Experiments are running on the dataset StanfordCars
* You may follow the data preparation guide [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).

## Running the experiments
*modify the configuration files in apps
```
modify the item "dataset_dir:" to your dataset 

```
*Train the model without compression and get a initialization_model.pt
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




## Performance verification with switchable training weights
The performance of the switchable training weights is similar to the performance of the separately trained weights.

You can download the individually trained weights by clicking on [this link](https://drive.google.com/drive/folders/1RGwLZUXjmtA3b_ok1v3Lcrwx-dGeidMU?usp=sharing).

After putting them under the current folder, you can test the performance of the downloaded weights with the following command.
```
python exp_test.py
```

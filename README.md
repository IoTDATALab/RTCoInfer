# RTCoInfer: Real-time Edge-Cloud Collaborative CNN Inference for Stream Analytics on Ubiquitous Images.

The implementation of paper : RTCoInfer: Real-time Edge-Cloud Collaborative CNN Inference for Stream Analytics on Ubiquitous Images. 

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


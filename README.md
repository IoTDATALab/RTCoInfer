# RTCoInfer: Real-time Edge-Cloud Collaborative CNN Inference for Stream Analytics on Ubiquitous Images.
RTCoInfer is an open-source framework for real-time collaborative CNN inference on ubiquitous images.  

Nowadays, emerging intelligent applications based on accurate and timely stream image analytics require real-time CNN inference of massive data continuously generated at the pervasive end devices. Due to the resource constraints, neither computing locally at end devices nor transmitting to remote servers is competent for computation-intensive CNN inference on large volume images in real-time. Therefore, collaborative inference, which conducts inference sequentially from the local device to the remote server with compressed intermediate inference data, is rapidly promoted.   

As a collaborative inference framework, RTCoInfer deploys a layer-wise partitioned CNN across the local device and the remote server, and conducts lightweight computation of a few initial CNN layers at the local device to compress intermediate inference data, then uploads the compressed intermediate inference data to the remote server to complete the computation-intensive inference and gets final results. For a stable network, the user can select a fixed partition and compression scheme considering the inference requirements.   
Moreover, considering the dynamic network in practice and the quality of real-time service (e.g., inference accuracy and responsiveness), RTCoInfer proposes the SWitchable-CNN (SW-CNN, i.e., a flexible CNN transformed from a given CNN model with the same architecture, which integrates the CNNs with different compression rate-accuracy loss tradeoffs and can switch the compression rate at run-time) to provide the adaptation ability to network fluctuations, and a real-time compression rate controller based on the Model Predictive Control (MPC) to provide high responsiveness and accuracy.  

The general workflow is illustrated as the following figure, and more details can be found in the paper “RTCoInfer: Real-time Collaborative CNN Inference for Stream Analytics on Ubiquitous Images”.

![img](assets/img/main.png)

## Table of Contents
  - [Table of Contents](#table-of-contents)
  - [Install](#install)
  - [Data Preparation](#data-preparation)
  - [Running Steps](#running-steps)
  - [Ablation Experiments](#ablation-experiments)
  - [Benchmark Verificatios.](#benchmark-verificatios)
  - [More Implementations](#more-implementations)

## Install
Clone repo and install requirements.txt in a Python>=3.7.0 environment, including PyTorch>=1.3.1.
```
git clone https://github.com/IoTDATALab/RTCoInfer.git  # clone
cd RTCoInfer
pip install -r requirements.txt  # install
```
## Data Preparation
* modify the configuration files in apps
```
modify the item "dataset_dir:" to your dataset 
```
* The current project takes the dataset StanfordCars as an example, and you may follow the data preparation guide [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).

## Running Steps
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

## Ablation Experiments
To demonstrate the effectiveness of Switchable CNN and analyze the contribution of each training technique (i.e., initialization(I), distillation(D) and clustering(C) techniques), we give the following ablation experiment results. Here, we take the accuracies of the models that are individually compressed under all compression setups as the baseline, and compare the accuracy losses of the SW-CNNs under the
same compression setups, where the SW-CNNs are trained by different combinations of these three training techniques.
Ideally, SW-CNN should achieve a similar accuracy with the individual compression method among all compression setups. Here, we represent the Initialization as “I”, the Distillation as“D”, and the Clustering as “C”. Then an SW-CNN trained with only Initialization can be represented as “I-noD-noC”. Finally, the accuracy losses of SW-CNNs compared with the individually compressed models are illustrated in the following figure, and details are given in the following table.
<div align="center">

<img src = assets/img/results.png width=60% />  

</div>

<div align="center">

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

The accuracy loss is the difference of top1 error between the SW-CNN and the model that is compressed individually under the same compression setup.
Obviously, under the same compression setup, the SW-CNN trained by I-D-S achieves the almost same performance to the model that is compressed individually (e.g., maximum difference is no more than 0.013).
Moreover, SWCNNs trained without Initialization (I) suffer large accuracy loss among all compression setups; SW-CNNs trained with Initialization (I) but without Distillation (D) can not recover the accuracy under small compression rates, thus they suffer a considerable maximum accuracy loss; SW-CNNs trained with Initialization (I) and Distillation (D) achieves the similar accuracy loss with I-D-S, while SW-CNNs trained with Clustering (C) reduces much training overhead.


## Benchmark Verificatios.
The accuracy of the SW-CNN is expected to be similar to the accuracy of the CNN that are compressed individually under the same compression setup. Therefore, the individually compressed CNNs under all compression setups are available as the benchmark for verifying the effectiveness of SW-CNN. You can download the weights of individually compressed CNNs under all compression setups by clicking on this [link](https://drive.google.com/drive/folders/1RGwLZUXjmtA3b_ok1v3Lcrwx-dGeidMU?usp=sharing).

After putting them under the current folder, you can test the performance of the downloaded weights with the following command.
```
python exp_test.py
```
Moreover, we also give the implementation of the state-of-the-art collaborative inference methods (i.e., SPINN and CLIO) for further comparison. 

<div align="center">

|CLIO|SPINN|
|:-:|:-:|
|Enabling automatic compilation of deep learning pipelines across IoT and Cloud. |Synergistic progressive inference of neural networks over device and cloud.
Original paper [link](https://dl.acm.org/doi/pdf/10.1145/3372224.3419215). |Original paper [link](https://dl.acm.org/doi/pdf/10.1145/3372224.3419194). 
The [implementation](https://github.com/IoTDATALab/RTCoInfer/tree/main/CLIO).  |The [implementation](https://github.com/IoTDATALab/RTCoInfer/tree/main/SPINN). |

</div>

## More Implementations
Besides the traditional CNN models like MobileNet and ResNet, the RTCoInfer can also be implemented on the Transform models containing CNN layers. Here, we take the MobileViT model (MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer) as an example, and the implementation is available by clicking on this [link](https://github.com/IoTDATALab/RTCoInfer/tree/main/MobileViT). 

# SPINN: synergistic progressive inference of neural networks over device and cloud

The implementation of paper : SPINN: synergistic progressive inference of neural networks over device and cloud.
![img_spinn](../assets/img/spinn.png)

## Data Resource
* Experiments are running on the dataset StanfordCars
* You may follow the data preparation guide [here](https://ai.stanford.edu/~jkrause/cars/car_dataset.html).
  
## Model Structure
View the model structure
```
./model.txt
```

## Running the experiments
* Train the model
```
python train.py app:'./apps/train_us_mobilenet_v2_train_val.yml'
```
* Test the model
```
python entropy.py app:'./apps/test_us_mobilenet_v2_train_val.yml'
```
* Change the test entropy
```
add the configuration "z_entropy: {number_you_want}" in test_us_mobilenet_v2_train_val.yml
```

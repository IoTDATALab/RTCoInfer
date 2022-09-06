import os
import yaml

width = [0.03125,0.0625,0.09375,0.1250,0.15625, 0.1875, 0.21875, 0.2500, 0.5000, 1.0]
quant = [255,15,3,1]
gpu = 0

yml_path = './apps/test.yml'
def set_state(state): 
    with open(yml_path) as f: 
        doc = yaml.load(f,Loader=yaml.FullLoader) 
    doc['train_only_one'] = state 
    doc['test_model'] = f'./mobilenet_v2_weights/quant{state[0][1]}/{state}.pt'
    with open(yml_path, 'w') as f: 
        yaml.dump(doc, f) 

for i in range(len(quant)):
    for j in range(len(width)):
        tp = [[width[j],quant[i]]]
        set_state(tp)
        with open(yml_path) as f: 
            doc = yaml.safe_load(f) 
            print(doc['train_only_one']) 
        os.system(f'python Individual_test.py app:apps/test.yml {gpu}')


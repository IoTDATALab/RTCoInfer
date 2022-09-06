import os
import yaml

width = [0.0625, 0.125, 0.25, 0.5, 1.0]
quant = [255,15,3,1]
gpu = 0

yml_path = './apps/test.yml'
def set_state(state): 
    with open(yml_path) as f: 
        doc = yaml.load(f,Loader=yaml.FullLoader) 
    doc['train_only_one'] = state 
    doc['test_model'] = f'./mobilevit_weights/quant{state[0][1]}/{state}.pt'
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


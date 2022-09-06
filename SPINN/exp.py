import os
import yaml

yml_path = './apps/test_us_mobilenet_v2_train_val.yml'
def set_state(state): 
    with open(yml_path) as f: 
        doc = yaml.load(f,Loader=yaml.FullLoader) 

    doc['z_entropy'] = state 

    with open(yml_path, 'w') as f: 
        yaml.dump(doc, f) 

z_entropy =  [0.5,0.25,0.4,0.6,0.7,0.46,0.45,0.3,0.35,0.335]
# 0.5 置信度阈值

    # 6993上云数量
    # 8041总共数量
    # 0.173平均精度
# 调整entropy，记录上传量和最后精度。

	#0.5-6993/8041-0.173 
	#0.25-4862/8041-0.323 
	#0.4-6471/8041-0.204 
	#0.6-7358/8041-0.156
	#0.7-7594/8041-0.148 1%
	#0.46-6823/8041-0.182
	#0.45-6766/8041-0.186 5%
	#0.3-5552/8041-0.269
	#0.35-6067-0.231 10%
	#0.335-5925-0.242 10%
for idx,value in enumerate(z_entropy):
    set_state(value)
    with open(yml_path) as f: 
        doc = yaml.safe_load(f) 
        print(doc['z_entropy']) 
    with open('res.txt','a+') as f:
        f.write(f'{value}')
        f.write('\n')
    os.system(f'python entropy.py app:{yml_path}')

import os

# for train
root_path=os.getcwd()
train_dir = f'{root_path}/input/data/train'
train_img_dir = f'{train_dir}/images'
df_path = f'{train_dir}/train.csv'

# for test
test_dir = f'{root_path}/input/data/eval'
test_img_dir = f'{test_dir}/images'
from tqdm import tqdm
import urllib.request
from pathlib import Path

data_location = 'https://gin.g-node.org/robintibor/high-gamma-dataset/raw/master/data/'

files = [str(i)+'.mat' for i in range(1,15)]
train_files = [(data_location + 'train/' + file, 'train/' + file) for file in files]
test_files = [(data_location + 'test/' + file, 'test/' + file) for file in files]



def load_and_store(file_list):
    for online_path, local_path in tqdm(file_list, desc='downloading'):
        urllib.request.urlretrieve(online_path, local_path)

print('dowloading... this will take a while. Please be patient.')
Path("train/").mkdir(exist_ok=True)
load_and_store(train_files)
print('downloading test data.')
Path("test/").mkdir(exist_ok=True)
load_and_store(test_files)

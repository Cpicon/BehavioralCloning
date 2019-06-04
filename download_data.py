from urllib.request import urlretrieve
import os
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from zipfile import ZipFile

def download_set(url, file):
    if not os.path.isfile(file):
        print("Download file... " + file + " ...")
        urlretrieve(url,file)
        print("File downloaded")

download_set('https://s3.amazonaws.com/video.udacity-data.com/topher/2016/December/584f6edd_data/data.zip','data.zip') 

print("All the files are downloaded")

def uncompress_data(dir,name):
    if(os.path.isdir(name)):
        print('Data extracted')
    else:
        with ZipFile(dir) as zipf:
            print('extracting.....')
            zipf.extractall('data')
            print('Data extracted')
uncompress_data('data.zip','data')


print('All files downloaded and extracted')

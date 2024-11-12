import requests
import tarfile
import os
import gdown

url = 'https://drive.google.com/uc?id=1hbzc_P1FuxMkcabkgn9ZKinBwW683j45' 
target_path = './data/CUB_200_2011/CUB-200-2011.tgz' 

gdown.download(url, target_path, quiet=False)
 
tar = tarfile.open(target_path, "r:gz")
tar.extractall(path='./data')
tar.close()
print("CUB downloaded")
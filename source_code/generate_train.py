import os
import random
path = r"E:/Python/TF/raccoon_dataset/annotations"
path_list = os.listdir(path)
path = [i.replace('.xml', '') for i in path_list]
random.shuffle(path)
with open(r"E:/Python/TF/raccoon_dataset/train.txt", 'w') as f:
    for p in path[:160]:
        f.write(p+'\n')
with open(r"E:/Python/TF/raccoon_dataset/val.txt", 'w') as f:
    for p in path[160:]:
        f.write(p+'\n')
import os
import random
import shutil
data_dir = "./data"
classes = os.listdir(data_dir)

for cls in classes:
   
    source_dir = os.path.join(data_dir, cls)
    img_paths = os.listdir(source_dir)
    num_train = int(0.8*len(img_paths))
    train_paths = random.sample(img_paths, num_train)
    #val_paths = [path if path not in train_paths for path in img_paths]
    val_paths = [path for path in img_paths if path not in train_paths]

    dest_dir_train = os.path.join("new_data", "train", cls)
    if not os.path.exists(dest_dir_train):
        os.makedirs(dest_dir_train)
    for path in train_paths:
        shutil.copy(os.path.join(source_dir,path), dest_dir_train)

    dest_dir_val = os.path.join("new_data", "val", cls)
    if not os.path.exists(dest_dir_val):
        os.makedirs(dest_dir_val)
    for path in val_paths:
        shutil.copy(os.path.join(source_dir,path), dest_dir_val)


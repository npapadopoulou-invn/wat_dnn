import os
import shutil

source = "/home/npapadopoulou/snippets_folder/simple2_snippets_ped150_rej10/test"  # main fol
val_users = ["User416", "User249"]  # validation users only

os.makedirs(os.path.join(os.path.dirname(source), "validation"), exist_ok=True)
os.makedirs(os.path.join(os.path.dirname(source), "test_split"), exist_ok=True)

for name in os.listdir(source):
    path = os.path.join(source, name)

    if os.path.isdir(path):
        for user in val_users:
            if user in name:
                validation_dir = os.path.join(os.path.dirname(source), "validation", name)
                shutil.copytree(path, validation_dir)
                break
        else:
            test_split_dir = os.path.join(os.path.dirname(source), "test_split", name)
            shutil.copytree(path, test_split_dir)

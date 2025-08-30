import os
import shutil

files = os.listdir('./dataset')

for file in files:
    if os.path.isfile(os.path.join('./dataset', file)):
        continue
    if len([i for i in os.listdir(os.path.join('./dataset',file)) if i.startswith('geetest_answer')]) == 0:
        shutil.rmtree(os.path.join('./dataset', file))
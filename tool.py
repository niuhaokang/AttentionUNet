import shutil
import os
from PIL import Image

source_dir = './result'
target_dir = '/data/haokang/Semi-Supervisied-DataSet/sketch/experiment/'

if not os.path.exists(target_dir):
    os.mkdir(target_dir)

shutil.move(source_dir,os.path.join(target_dir,'result_patchnum1024'))

# source_dir = '/data/haokang/Semi-Supervisied-DataSet/sketch/test'
# target_dir = '/data/haokang/Semi-Supervisied-DataSet/sketch/test_'
#
# if not os.path.exists(target_dir):
#     os.mkdir(target_dir)
#
# imgs = [os.path.join(source_dir, img) for img in os.listdir(source_dir)]
#
# for img in imgs:
#     save_name = os.path.join(target_dir, img.split('/')[-1])
#     I = Image.open(img)
#     L = I.convert('L')
#     L.save(save_name)

import os
import glob
import argparse

parser = argparse.ArgumentParser(description='data collection -> rename')
parser.add_argument('--source_img', type=str)
parser.add_argument('--source_ano', type=str)
parser.add_argument('--dest_img', type=str)
parser.add_argument('--dest_ano', type=str)
parser.add_argument('--start_index', type=int)

opt = parser.parse_args()

img_path_ = opt.source_img + '/'
ano_path_ = opt.source_ano + '/'

dest_img_path = opt.dest_img + '/'
dest_ano_path = opt.dest_ano + '/'

img_path = img_path_ + '*.jpg'

img = glob.glob(img_path)

count = opt.start_index

for tmp in img:
    tmp_label = tmp.split('/')[-1].split('.')[0]
    print(os.path.join(ano_path_, tmp_label + '.xml'))
    os.rename(tmp, os.path.join(dest_img_path, str(count) + '.jpg'))
    print(tmp_label)
    os.rename(os.path.join(ano_path_, tmp_label + '.xml'), os.path.join(dest_ano_path, str(count) + '.xml'))
    count += 1

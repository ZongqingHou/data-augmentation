import os
import glob
import random

def shuffle_datasets(img_path, ano_path, dest_img_path, dest_ano_path):
    img_collections = glob.glob(img_path)

    index_collections = [tmp for tmp in range(len(img_collections))]
    
    for tmp in img_collections:
        tmp_index = random.choice(index_collections)
        tmp_label = tmp.split('/')[-1].split('.')[0]
        os.rename(tmp, os.path.join(dest_img_path, str(tmp_index) + '.jpg'))
        os.rename(os.path.join(ano_path, tmp_label + '.xml'), os.path.join(dest_ano_path, str(tmp_index) + '.xml'))

        index_collections.remove(tmp_index)

if __name__ == '__main__':
    shuffle_datasets('/home/extension/datasets/bullect_collection/backup/trainning/2/images/*.jpg',
                     '/home/extension/datasets/bullect_collection/backup/trainning/2/xml/',
                     '/home/extension/datasets/bullect_collection/backup/trainning/3/images/',
                     '/home/extension/datasets/bullect_collection/backup/trainning/3/xml/')
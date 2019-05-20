import os
import argparse
import threading
import random

import augs
import utils

function_list = dir(augs)

class DataAugThread(threading.Thread):
    def __init__(self, img_list, configs):
        super(DataAugThread, self).__init__()

        self.img_list = img_list
        self.configs = configs

    def run(self):
        for tmp_img in self.img_list:
            pass
        

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='DataAug')
	parser.add_argument('--img_source_path', type=str, default='/home/hdd/Desktop/dataset/yt20190404')
	parser.add_argument('--ano_source_path', type=str, default='/home/hdd/Desktop/dataset/bt20190405')
	parser.add_argument('--img_dest_path', type=str, default='/home/hdd/Desktop/dataset/tmp_img')
	parser.add_argument('--ano_dest_path', type=str, default='/home/hdd/Desktop/dataset/tmp_ano')
	parser.add_argument('--start_name', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
	args = parser.parse_args()

	source_pic_root_path = args.img_source_path
	source_xml_root_path = args.ano_source_path
	img_root_path = args.img_dest_path
	aug_root_path = args.ano_dest_path

    start_index = args.start_name
    num_workers = args.workers

    img_list = utils.read_files(source_pic_root_path)
    config = utils.load_json()

    tmp_len = len(img_list) / num_workers
    
    thread_pool = []
    for tmp in range(num_workers):
        if tmp + 1 == num_workers:
            tmp_thread = DataAugThread(img_list[tmp*tmp_len:], config)
        else:
            tmp_thread = DataAugThread(img_list[tmp*tmp_len:(tmp+1)*tmp_len], config)
        
        thread_pool.append(tmp_thread)
    
    for tmp_thread in thread_pool:
        tmp_thread.start()
        tmp_thread.join()
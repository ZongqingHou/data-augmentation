import os
import cv2
import copy
import argparse
import threading
import random

from augs import DataAugmentation as da
import utils

function_list = [tmp for tmp in dir(da) if '__' not in tmp]

class DataAugThread(threading.Thread):
	def __init__(self, img_list, src_json_path_root, src_xml_path_root, dest_img_path_root, dest_json_path_root, dest_xml_path_root):
		super(DataAugThread, self).__init__()

		self.img_list = img_list
		self.src_json_path_root = src_json_path_root + '/' if src_json_path_root[-1] != '/' else src_json_path_root
		self.src_xml_path_root = src_xml_path_root + '/' if src_xml_path_root[-1] != '/' else src_xml_path_root
		self.dest_img_path_root = dest_img_path_root + '/' if dest_img_path_root[-1] != '/' else dest_img_path_root
		self.dest_json_path_root = dest_json_path_root + '/' if dest_json_path_root[-1] != '/' else dest_json_path_root
		self.dest_xml_path_root = dest_xml_path_root + '/' if dest_xml_path_root[-1] != '/' else dest_xml_path_root

	def run(self):
		global start_index, function_list

		for tmp_img in self.img_list:
			for tmp_func in function_list:
				img, ano_points, ano_box = getattr(da, tmp_func)(img, ano_points, ano_box)
				utils.save(start_index, img, ano_points, ano_box, self.dest_img_path_root, self.dest_json_path_root, self.dest_xml_path_root)
				start_index += 1


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='DataAug')
	parser.add_argument('--img_source_path', type=str, default='/home/hdd/hdD_Git/FCOS/datasets/bullet/src/images')
	parser.add_argument('--json_source_path', type=str, default='/home/hdd/hdD_Git/FCOS/datasets/bullet/src/json')
	parser.add_argument('--ano_source_path', type=str, default='/home/hdd/hdD_Git/FCOS/datasets/bullet/src/xml')
	parser.add_argument('--img_dest_path', type=str, default='/home/hdd/hdD_Git/FCOS/datasets/bullet/dest/images')
	parser.add_argument('--json_dest_path', type=str, default='/home/hdd/hdD_Git/FCOS/datasets/bullet/dest/json')
	parser.add_argument('--ano_dest_path', type=str, default='/home/hdd/hdD_Git/FCOS/datasets/bullet/dest/xml')
	parser.add_argument('--start_name', type=int, default=0)
	parser.add_argument('--workers', type=int, default=4)
	args = parser.parse_args()

	source_pic_root_path = args.img_source_path
	source_json_root_path = args.json_source_path
	source_xml_root_path = args.ano_source_path
	img_root_path = args.img_dest_path
	json_root_path = args.json_dest_path
	aug_root_path = args.ano_dest_path

	start_index = args.start_name
	num_workers = args.workers

	img_list = utils.read_files(source_pic_root_path)

	tmp_len = int(len(img_list) / num_workers)
	
	thread_pool = []
	for tmp in range(num_workers):
		if tmp + 1 == num_workers:
			tmp_thread = DataAugThread(img_list[tmp*tmp_len:], source_json_root_path, source_xml_root_path, img_root_path, json_root_path, aug_root_path)
		else:
			tmp_thread = DataAugThread(img_list[tmp*tmp_len:(tmp+1)*tmp_len], source_json_root_path, source_xml_root_path, img_root_path, json_root_path, aug_root_path)
		
		thread_pool.append(tmp_thread)
	
	for tmp_thread in thread_pool:
		tmp_thread.start()
		tmp_thread.join()
# -*- coding=utf-8 -*-
import cv2
import json
import glob
import numpy as np
import base64
import random
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as DOC

offset_flag = [True, False]

def intersection(default_index, list_collections):
	if default_index == len(list_collections) - 1:
		return list_collections[0:default_index]
	else:
		return list_collections[0:default_index] + list_collections[default_index+1:]


def save(index, img, ano_points, ano_box, dest_img_path_root, dest_json_path_root, dest_xml_path_root, debug=''):
	cv2.imwrite(dest_img_path_root + debug + '%s.jpg' %index, img)
	with open(dest_img_path_root + debug + '%s.jpg' %index, 'rb') as buffer:
		img_data = buffer.read()
		encodestr = base64.b64encode(img_data)
	
	ano_points["imageData"] = str(encodestr)[2:-1]
	ano_points["imageHeight"] = img.shape[0]
	ano_points["imageWidth"] = img.shape[1]
	ano_points["imagePath"] = dest_img_path_root + debug + '%s.jpg' %index

	save_json(ano_points, dest_json_path_root + debug + '%s.json' %index)
	generate_xml("images", ano_box, img.shape, dest_xml_path_root + debug + '%s.xml' %index)

def name(string):
	return string.split('/')[-1].split('.')[0]
	
# 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(xml_path):
	'''
	输入：
		xml_path: xml的文件路径
	输出：
		从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
	'''
	tree = ET.parse(xml_path)
	root = tree.getroot()
	objs = root.findall('object')
	coords = list()
	for ix, obj in enumerate(objs):
		name = obj.find('name').text
		box = obj.find('bndbox')
		x_min = int(box[0].text)
		y_min = int(box[1].text)
		x_max = int(box[2].text)
		y_max = int(box[3].text)
		coords.append([x_min, y_min, x_max, y_max, name])
	return coords

#将bounding box信息写入xml文件中, bouding box格式为[[x_min, y_min, x_max, y_max, name]]
def generate_xml(img_name,coords,img_size, output_path):
	'''
	输入：
		img_name：图片名称，如a.jpg
		coords:坐标list，格式为[[x_min, y_min, x_max, y_max, name]]，name为概况的标注
		img_size：图像的大小,格式为[h,w,c]
		out_root_path: xml文件输出的根路径
	'''
	doc = DOC.Document()  # 创建DOM文档对象

	annotation = doc.createElement('annotation')
	doc.appendChild(annotation)

	title = doc.createElement('folder')
	title_text = doc.createTextNode('Tianchi')
	title.appendChild(title_text)
	annotation.appendChild(title)

	title = doc.createElement('filename')
	title_text = doc.createTextNode(img_name)
	title.appendChild(title_text)
	annotation.appendChild(title)

	source = doc.createElement('source')
	annotation.appendChild(source)

	title = doc.createElement('database')
	title_text = doc.createTextNode('The Tianchi Database')
	title.appendChild(title_text)
	source.appendChild(title)

	title = doc.createElement('annotation')
	title_text = doc.createTextNode('Tianchi')
	title.appendChild(title_text)
	source.appendChild(title)

	size = doc.createElement('size')
	annotation.appendChild(size)

	title = doc.createElement('width')
	title_text = doc.createTextNode(str(img_size[1]))
	title.appendChild(title_text)
	size.appendChild(title)

	title = doc.createElement('height')
	title_text = doc.createTextNode(str(img_size[0]))
	title.appendChild(title_text)
	size.appendChild(title)

	title = doc.createElement('depth')
	title_text = doc.createTextNode(str(img_size[2]))
	title.appendChild(title_text)
	size.appendChild(title)

	for coord in coords:

		object = doc.createElement('object')
		annotation.appendChild(object)

		title = doc.createElement('name')
		title_text = doc.createTextNode(coord[4])
		title.appendChild(title_text)
		object.appendChild(title)

		pose = doc.createElement('pose')
		pose.appendChild(doc.createTextNode('Unspecified'))
		object.appendChild(pose)
		truncated = doc.createElement('truncated')
		truncated.appendChild(doc.createTextNode('1'))
		object.appendChild(truncated)
		difficult = doc.createElement('difficult')
		difficult.appendChild(doc.createTextNode('0'))
		object.appendChild(difficult)

		bndbox = doc.createElement('bndbox')
		object.appendChild(bndbox)
		title = doc.createElement('xmin')
		title_text = doc.createTextNode(str(int(float(coord[0]))))
		title.appendChild(title_text)
		bndbox.appendChild(title)
		title = doc.createElement('ymin')
		title_text = doc.createTextNode(str(int(float(coord[1]))))
		title.appendChild(title_text)
		bndbox.appendChild(title)
		title = doc.createElement('xmax')
		title_text = doc.createTextNode(str(int(float(coord[2]))))
		title.appendChild(title_text)
		bndbox.appendChild(title)
		title = doc.createElement('ymax')
		title_text = doc.createTextNode(str(int(float(coord[3]))))
		title.appendChild(title_text)
		bndbox.appendChild(title)

	# 将DOM对象doc写入文件
	f = open(output_path,'w')
	f.write(doc.toprettyxml(indent = ''))
	f.close()

def load_json(path='./aug_config.json'):
	with open(path, 'r') as file_buffer:
		load_dict = json.load(file_buffer)

	return load_dict

def save_json(load_dict, path='./aug_config.json'):
	with open(path, 'w') as file_buffer:
		json.dump(load_dict, file_buffer)

def read_files(img_path):
	img_path = img_path if img_path[-1] == '/' else img_path + '/'
	img_path += '*'

	img_list = glob.glob(img_path)
	return img_list

def max_min(points):
	x_collections = []
	y_collections = []

	for tmp_coord in points:
		x_collections.append(tmp_coord[0])
		y_collections.append(tmp_coord[1])

	return [min(x_collections), min(y_collections), max(x_collections), max(y_collections)]

def random_offerset(max_min_coords, img_width, img_height):
	width_flag = random.choice(offset_flag)
	height_flag = random.choice(offset_flag)

	width_offset = random.randint(1, img_width - max_min_coords[-2]) if width_flag else -random.randint(1, max_min_coords[0])
	height_offset = random.randint(1, img_height - max_min_coords[-1]) if height_flag else -random.randint(1, max_min_coords[1])

	return width_offset, height_offset

def point_offset(points, offset):
	for tmp_coord in points:
		tmp_coord[0] += offset[0]
		tmp_coord[1] += offset[1]

	return points

def copy_img(img, src, dest):
	for index, tmp_coord in enumerate(src):
		img[dest[1]:dest[3], dest[0]:dest[2]] = img[src[1]:src[3], src[0]:src[2]]
		# img[dest[index][1]][dest[index][0]] = img[tmp_coord[1]][tmp_coord[0]]

def mat_product(matrix, points):
	result_list = []
	for tmp_coord in points:
		tmp_point = np.dot(matrix, np.array(tmp_coord + [1])).astype(np.int).tolist()
		result_list.append(tmp_point)

	return result_list

def mat_minus(min_collections, points):
	result_list = []
	for tmp_coord in points:
		tmp_point = [tmp_coord[0] - min_collections[0], tmp_coord[1] - min_collections[1]]
		result_list.append(tmp_point)

	return result_list

def flip(w, h, horizon, points):
	result_list = []
	for tmp_coord in points:
		if horizon:
			tmp_point = [w - tmp_coord[0], tmp_coord[1]]
		else:
			tmp_point = [tmp_coord[0], h - tmp_coord[1]]
		result_list.append(tmp_point)

	return result_list

def augmentation(start_index, module, function_name, img, points, box, thread_module):	
	img, points, box = getattr(da, function_name)(img, points, box)
	utils.save(start_index, img, ano_points, ano_box, thread_module.dest_img_path_root, thread_module.dest_json_path_root, thread_module.dest_xml_path_root)
	start_index += 1
	return start_index, img, points, box

def show_pic(img, bboxes=None):
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),3) 
		
    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200,800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

if __name__ == "__main__":
	import io
	from PIL import Image
	dest = "/home/hdd/git/data-augmentation/test_data/dest/pairs/"
	json_root = "/home/hdd/git/data-augmentation/test_data/src/json/*.json"
	json_files = glob.glob(json_root)

	index_name = 0

	for tmp_json in json_files:
		with open(tmp_json, 'r') as file_buffer:
			load_dict = json.load(file_buffer)

		image_buffer = io.BytesIO(base64.b64decode(load_dict["imageData"]))
		image = Image.open(image_buffer)

		xml_ano = []

		for tmp_coord in load_dict["shapes"]:
			# if tmp_coord["label"] == "boundary" or tmp_coord["label"] == "target":
			# 	continue

			bbox = max_min(tmp_coord["points"])
			bbox.append(tmp_coord["label"])
			xml_ano.append(bbox)
		
		image.save(dest + "images/%s.jpg" %index_name)

		img_size = [image.size[1], image.size[0], 3]

		generate_xml(str(index_name), xml_ano, img_size, dest + "ano/%s.xml" %index_name)

		index_name += 1
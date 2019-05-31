import cv2
import random
import numpy as np

class Perspect:
	def perspect(img, json_file, xml_file):
		label = xml_file[-1]

		polygon = json_file["shapes"]
		img_width = json_file["imageWidth"]
		img_height = json_file["imageHeight"]
		base_information = {"fill_color": None, "shape_type": "polygon", "label": label, "line_color": None}

		tmp_points = []
		for tmp_ano in polygon:
			tmp_max_min = utils.max_min(tmp_ano["points"])
			width_offset, height_offset = utils.random_offerset(tmp_max_min, img_width, img_height)
			points = copy.deepcopy(tmp_ano["points"])
			points = utils.point_offset(points, [width_offset, height_offset])
			tmp_xml = utils.max_min(points) + ["bullet"]
			base_information["points"] = points
			tmp_points.append(copy.deepcopy(base_information))
			ano_box.append(tmp_xml)
			utils.copy_img(img, tmp_max_min, tmp_xml)

		polygon += tmp_points
		return img, ano_points, ano_box

	def matrix(src_coord, dest_coord):
		prspct_org = np.float32(src_coord)
		prspct_dst = np.float32(dest_coord)

		perspect = cv2.getPerspectiveTransform(prspct_org, prspct_dst)

		return perspect
	
	def transform_img(img, matrix, dest_size):
		return cv2.warpPerspective(img, matrix, dest_size)
		
	def transform_coord(point, matrix):
		src_point = np.array(point + [1])
		tmp_result = np.dot(matrix, src_point)

		return tmp_result[0] / tmp_result[2], tmp_result[1] / tmp_result[2]

if __name__ == "__main__":
	import sys
	sys.path.append("../")

	import glob
	import utils
	import argparse

	parser = argparse.ArgumentParser(description='Transform')
	parser.add_argument('--img_source_path', type=str, default='/home/hdd/hdD_Git/FCOS/datasets/bullet/src/images')
	parser.add_argument('--json_source_path', type=str, default='/home/hdd/hdD_Git/FCOS/datasets/bullet/src/json')
	parser.add_argument('--ano_source_path', type=str, default='/home/hdd/hdD_Git/FCOS/datasets/bullet/src/xml')
	parser.add_argument('--img_dest_path', type=str, default='/home/hdd/hdD_Git/FCOS/datasets/bullet/dest/images')
	parser.add_argument('--json_dest_path', type=str, default='/home/hdd/hdD_Git/FCOS/datasets/bullet/dest/json')
	parser.add_argument('--ano_dest_path', type=str, default='/home/hdd/hdD_Git/FCOS/datasets/bullet/dest/xml')
	parser.add_argument('--start_name', type=int, default=0)

	args = parser.parse_args()

	img_path = args.img_source_path + '/' if args.img_source_path[-1] != '/' else args.img_source_path

	src_json_path_root = args.json_source_path + '/' if args.json_source_path[-1] != '/' else args.json_source_path
	src_xml_path_root = args.ano_source_path + '/' if args.ano_source_path[-1] != '/' else args.ano_source_path
	dest_img_path_root = args.img_dest_path + '/' if args.img_dest_path[-1] != '/' else args.img_dest_path
	dest_json_path_root = args.json_dest_path + '/' if args.json_dest_path[-1] != '/' else args.json_dest_path
	dest_xml_path_root = args.ano_dest_path + '/' if args.ano_dest_path[-1] != '/' else args.ano_dest_path

	img_list = glob.glob(img_path + "*.jpg")

	for tmp_img in img_list:
		name = utils.name(tmp_img)

		img = cv2.imread(tmp_img)
		ano_points = utils.load_json(src_json_path_root + name + ".json")
		ano_box = utils.parse_xml(src_xml_path_root + name + ".xml")

		height, width, _ = img.shape
		src_point = [[0, height], [width, height], [100, 0], [width - 100, 0]]
		dest_point = [[0, height + 100], [width + 100, height + 100], [0, 0], [width + 100, 0]]

		matrix = Perspect.matrix(src_point, dest_point)
		tmp_img = Perspect.transform_img(img, matrix, (int(width + 100), int(height + 100)))

		print(ano_box)
		tmp_ = [Perspect.transform_coord(tmp[:2], matrix) + Perspect.transform_coord(tmp[2:-1], matrix) for tmp in ano_box]
		print(tmp_)
		utils.show_pic(tmp_img, tmp_)

		cv2.imshow('d', img)
		cv2.imshow('t', tmp_img)
		cv2.waitKey(0)
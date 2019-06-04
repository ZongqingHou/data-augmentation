import cv2
import random
import numpy as np

class Perspect:
	def perspect(img, json_file, xml_file):
		polygon = json_file["shapes"]
		width = json_file["imageWidth"]
		height = json_file["imageHeight"]

		width_offset = random.randint(int(1 / 36 * width), int(1 / 16 * width))
		height_offset = random.randint(int(1 / 36 * width), int(1 / 16 * height))

		src_point = [[0, height], [width, height], [width_offset, 0], [width - width_offset, 0]]
		dest_point = [[0, height + height_offset], [width, height + height_offset], [0, 0], [width, 0]]
		matrix = Perspect.matrix(src_point, dest_point)

		trans_img = Perspect.transform_img(img, matrix, (width, height + height_offset))

		for tmp_ano in polygon:
			tmp_ano["points"] = Perspect.json_parse_points(matrix, tmp_ano["points"])

		bbox = Perspect.xml_parse_points(matrix, xml_file)
			
		return trans_img, json_file, bbox

	def json_parse_points(matrix, points):
		return [Perspect.transform_coord(tmp_point, matrix) for tmp_point in points]

	def xml_parse_points(matrix, bbox):
		return [Perspect.transform_coord(tmp[:2], matrix) + Perspect.transform_coord(tmp[2:-1], matrix) + [tmp[-1]] for tmp in bbox]

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

		return [tmp_result[0] / tmp_result[2], tmp_result[1] / tmp_result[2]]

if __name__ == "__main__":
	import sys
	sys.path.append("../")

	import glob
	import utils
	import argparse

	parser = argparse.ArgumentParser(description='Transform')
	parser.add_argument('--img_source_path', type=str, default='/home/hdd/hdD_Git/data-augmentation/for_test/dest/img')
	parser.add_argument('--json_source_path', type=str, default='/home/hdd/hdD_Git/data-augmentation/for_test/dest/json')
	parser.add_argument('--ano_source_path', type=str, default='/home/hdd/hdD_Git/data-augmentation/for_test/dest/xml')
	parser.add_argument('--img_dest_path', type=str, default='/home/hdd/hdD_Git/data-augmentation/for_test/tmp/img')
	parser.add_argument('--json_dest_path', type=str, default='/home/hdd/hdD_Git/data-augmentation/for_test/tmp/json')
	parser.add_argument('--ano_dest_path', type=str, default='/home/hdd/hdD_Git/data-augmentation/for_test/tmp/xml')
	parser.add_argument('--start_name', type=int, default=336)

	args = parser.parse_args()

	img_path = args.img_source_path + '/' if args.img_source_path[-1] != '/' else args.img_source_path

	src_json_path_root = args.json_source_path + '/' if args.json_source_path[-1] != '/' else args.json_source_path
	src_xml_path_root = args.ano_source_path + '/' if args.ano_source_path[-1] != '/' else args.ano_source_path
	dest_img_path_root = args.img_dest_path + '/' if args.img_dest_path[-1] != '/' else args.img_dest_path
	dest_json_path_root = args.json_dest_path + '/' if args.json_dest_path[-1] != '/' else args.json_dest_path
	dest_xml_path_root = args.ano_dest_path + '/' if args.ano_dest_path[-1] != '/' else args.ano_dest_path

	img_list = glob.glob(img_path + "*.jpg")
	start_name = args.start_name
	for tmp_img in img_list:
		name = utils.name(tmp_img)

		img = cv2.imread(tmp_img)
		ano_points = utils.load_json(src_json_path_root + name + ".json")
		ano_box = utils.parse_xml(src_xml_path_root + name + ".xml")

		p_img, p_json, p_bbox = Perspect.perspect(img, ano_points, ano_box)
		utils.save(start_name, p_img, p_json, p_bbox, dest_img_path_root, dest_json_path_root, dest_xml_path_root)
		start_name += 1
		
		# utils.show_pic(p_img, p_bbox)
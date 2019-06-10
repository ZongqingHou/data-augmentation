import sys
sys.path.append("../")

import utils

def add_target(img, ano_points, ano_box):
	polygon = ano_points["shapes"]
	img_width = ano_points["imageWidth"]
	img_height = ano_points["imageHeight"]
	base_information = {"fill_color": None, "shape_type": "polygon", "label": "bullet", "line_color": None}
	tmp_points = []
	for tmp_ano in polygon:
		if tmp_ano["label"] == "bullet":
			tmp_max_min = utils.max_min(tmp_ano["points"])
			width_offset, height_offset = utils.random_offerset(tmp_max_min, img_width, img_height)
			points = copy.deepcopy(tmp_ano["points"])
			points = utils.point_offset(points, [width_offset, height_offset])
			tmp_xml = utils.max_min(points) + ["bullet"]
			base_information["points"] = points
			tmp_points.append(copy.deepcopy(base_information))
			ano_box.append(tmp_xml)
			utils.copy_img(img, tmp_max_min, tmp_xml)
		else:
			pass

	polygon += tmp_points
	return img, ano_points, ano_box

if __name__ == "__main__":
	import cv2
	import glob
	import copy
	import argparse

	parser = argparse.ArgumentParser(description='Transform')
	parser.add_argument('--img_source_path', type=str, default='/home/extension/datasets/bullect_collection/backup/trainning/4/images')
	parser.add_argument('--json_source_path', type=str, default='/home/extension/datasets/bullect_collection/backup/trainning/4/json')
	parser.add_argument('--ano_source_path', type=str, default='/home/extension/datasets/bullect_collection/backup/trainning/4/xml')
	parser.add_argument('--img_dest_path', type=str, default='/home/extension/datasets/bullect_collection/backup/trainning/2/images')
	parser.add_argument('--json_dest_path', type=str, default='/home/extension/datasets/bullect_collection/backup/trainning/2/json')
	parser.add_argument('--ano_dest_path', type=str, default='/home/extension/datasets/bullect_collection/backup/trainning/2/xml')
	parser.add_argument('--start_name', type=int, default=29058)

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
		img = cv2.imread(tmp_img)
		ano_points = utils.load_json(src_json_path_root + utils.name(tmp_img) + '.json')
		ano_box = utils.parse_xml(src_xml_path_root + utils.name(tmp_img) + '.xml')
		img_more, points_more, box_more = add_target(copy.deepcopy(img),
													 copy.deepcopy(ano_points),
													 copy.deepcopy(ano_box))					

		img_more, points_more, box_more = add_target(img_more, points_more, box_more)
		utils.save(start_name, img_more, points_more, box_more, dest_img_path_root, dest_json_path_root, dest_xml_path_root)
		start_name += 1
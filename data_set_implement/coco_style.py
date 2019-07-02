import json

'''
{
	"images":[
		{
			"file_name": "000000397133.jpg",
		 	"height": 427,
		 	"width": 640,
		 	"id": 397133
		}
	]
	"annotations": [
		{
			"segmentation": [[]],
			"area": 702.1057499999998,
			"iscrowd": 0,
			"image_id": 289343,
			"bbox": [473.07,395.93,38.65,28.67],
			"category_id": 18,
			"id": 1768
		}
	]
}
'''
category_dict = {
	"bullet": 0,
	"boundary": 1,
	"target": 2
}

if __name__ == "__main__":
	import glob
	import argparse

	import sys
	sys.path.append("../")

	from utils import *

	parse = argparse.ArgumentParser("coco style")
	parse.add_argument("--src_path", type=str, default="/home/hdd/hdD_Git/data_augmentation/for_test/src/json")
	parse.add_argument("--dest_path", type=str, default="tt.json")

	args = parse.parse_args()

	src_json_path = args.src_path if args.src_path[-1] == "/" else args.src_path + "/"
	json_files = glob.glob(src_json_path + "*.json")

	coco_style = {}

	index_count = 0

	for tmp_file in json_files:
		with open(tmp_file, 'r', encoding="unicode_escape") as file_buffer:
			load_dict = json.load(file_buffer)
		
		file_id = int(tmp_file.split("/")[-1].split(".")[0])

		image_field = {
			"file_name": "%s.jpg" %file_id,
		 	"height": load_dict["imageHeight"],
		 	"width": load_dict["imageWidth"],
		 	"id": file_id
		}

		if "images" not in coco_style:
			coco_style["images"] = [image_field]
		else:
			coco_style["images"].append(image_field)

		for tmp_polygon in load_dict["shapes"]:
			bbox = max_min(tmp_polygon["points"])

			polygon_list = []

			for tmp in tmp_polygon["points"]:
				polygon_list.extend(tmp)

			annotations = {
				"segmentation": [polygon_list],
				"area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
				"iscrowd": 0,
				"image_id": file_id,
				"bbox":  [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]],
				"category_id": category_dict[tmp_polygon["label"]],
				"id": index_count
			}

			index_count += 1

			if "annotations" not in coco_style:
				coco_style["annotations"] = [annotations]
			else:
				coco_style["annotations"].append(annotations)

	with open(args.dest_path, 'w') as file_buffer:
		json.dump(coco_style, file_buffer)
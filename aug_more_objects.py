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
    import utils

    for tmp_img in self.img_list:
		img = cv2.imread(tmp_img)
		ano_points = utils.load_json(self.src_json_path_root + utils.name(tmp_img) + '.json')
		ano_box = utils.parse_xml(self.src_xml_path_root + utils.name(tmp_img) + '.xml')
		img_more, points_more, box_more = add_target(copy.deepcopy(img),
										 		     copy.deepcopy(ano_points),
										 			 copy.deepcopy(ano_box))
        
        img_more, points_more, box_more = getattr(da, tmp_func)(img_more, points_more, box_more)
		utils.save(start_index, img_more, points_more, box_more, self.dest_img_path_root, self.dest_json_path_root, self.dest_xml_path_root)
		start_index += 1
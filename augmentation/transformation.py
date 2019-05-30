import cv2
import numpy as np

class Perspect:
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

	import utils

	img = cv2.imread("/home/hdd/hdD_Git/data-augmentation/for_test/src/img/aa_0.jpg")
	ano_points = utils.load_json("/home/hdd/hdD_Git/data-augmentation/for_test/src/json/aa_0.json")
	ano_box = utils.parse_xml("/home/hdd/hdD_Git/data-augmentation/for_test/src/xml/aa_0.xml")

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
# img = cv2.imread(img_path)
# ano_points = utils.load_json(json_ano)
# ano_box = utils.parse_xml(xml_ano)
# cv2.imwrite(img_path[:-4] + "_more.jpg", img)
# utils.save_json(ano_points, json_ano[:-4] + "_more.json")
# utils.generate_xml("images", ano_box, img.shape, xml_ano[:-4] + "_more.xml")

# # cutout
# def _cutout(img, bboxes, length=100, n_holes=1, threshold=0.5):
# 	'''
# 	原版本：https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
# 	Randomly mask out one or more patches from an image.
# 	Args:
# 		img : a 3D numpy array,(h,w,c)
# 		bboxes : 框的坐标
# 		n_holes (int): Number of patches to cut out of each image.
# 		length (int): The length (in pixels) of each square patch.
# 	'''
# 	def cal_iou(boxA, boxB):
# 		'''
# 		boxA, boxB为两个框，返回iou
# 		boxB为bouding box
# 		'''
# 		# determine the (x, y)-coordinates of the intersection rectangle
# 		xA = max(boxA[0], boxB[0])
# 		yA = max(boxA[1], boxB[1])
# 		xB = min(boxA[2], boxB[2])
# 		yB = min(boxA[3], boxB[3])
# 		if xB <= xA or yB <= yA:
# 			return 0.0
# 		# compute the area of intersection rectangle
# 		interArea = (xB - xA + 1) * (yB - yA + 1)
# 		# compute the area of both the prediction and ground-truth
# 		# rectangles
# 		boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
# 		boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
# 		# compute the intersection over union by taking the intersection
# 		# area and dividing it by the sum of prediction + ground-truth
# 		# areas - the interesection area
# 		# iou = interArea / float(boxAArea + boxBArea - interArea)
# 		iou = interArea / float(boxBArea)
# 		# return the intersection over union value
# 		return iou
# 	# 得到h和w
# 	if img.ndim == 3:
# 		h,w,c = img.shape
# 	else:
# 		_,h,w,c = img.shape
# 	mask = np.ones((h,w,c), np.float32)
# 	for n in range(n_holes):
# 		chongdie = True    #看切割的区域是否与box重叠太多
# 		while chongdie:
# 			y = np.random.randint(h)
# 			x = np.random.randint(w)
# 			y1 = np.clip(y - length // 2, 0, h)    #numpy.clip(a, a_min, a_max, out=None), clip这个函数将将数组中的元素限制在a_min, a_max之间，大于a_max的就使得它等于 a_max，小于a_min,的就使得它等于a_min
# 			y2 = np.clip(y + length // 2, 0, h)
# 			x1 = np.clip(x - length // 2, 0, w)
# 			x2 = np.clip(x + length // 2, 0, w)
# 			chongdie = False
# 			for box in bboxes:
# 				if cal_iou([x1,y1,x2,y2], box) > threshold:
# 					chongdie = True
# 					break
# 		mask[y1: y2, x1: x2, :] = 0.
# 	# mask = np.expand_dims(mask, axis=0)
# 	img = img * mask
# 	return img
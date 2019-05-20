import copy

import os
import cv2
import skimage
import random
from skimage import io, filters, img_as_float, exposure
from skimage.morphology import disk
import skimage.filters.rank as sfr

import utils

def add_target(img_path, json_ano, xml_ano, base_offset=100):
	img = cv2.imread(img_path)
	ano_points = utils.load_json(json_ano)
	ano_box = utils.parse_xml(xml_ano)

	# point chain list
	polygon = ano_points["shapes"]
	img_width = ano_points["imageWidth"]
	img_height = ano_points["imageHeight"]

	base_information =     {
			"fill_color": None,
			"shape_type": "polygon",
			"label": "bullet",
			"line_color": None
		}

	tmp_points = []

	for tmp_ano in polygon:
		if tmp_ano["label"] == "bullet":
			tmp_max_min = utils.max_min(tmp_ano["points"])
			
			width_offset = random.randint(base_offset, img_width - tmp_max_min[-2])
			height_offset = random.randint(base_offset, img_height - tmp_max_min[-1])

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

	# return img, ano_points, ano_box
	cv2.imwrite(img_path[:-4] + "_more.jpg", img)
	utils.save_json(ano_points, json_ano[:-4] + "_more.json")
	utils.generate_xml("images", ano_box, img.shape, xml_ano[:-4] + "_more.xml")

def addNoise(img):
	'''
	输入:
		img:图像array
	输出:
		加噪声后的图像array,由于输出的像素是在[0,1]之间,所以得乘以255
	'''
	return random_noise(img, mode='gaussian', clip=True) * 255

# 调整亮度
def changeLight(img):
	# random.seed(int(time.time()))
	flag = random.uniform(0.5, 1) #flag>1为调暗,小于1为调亮
	return exposure.adjust_gamma(img, flag)

def darkness(img):
	# random.seed(int(time.time()))
	flag = random.uniform(1, 1.5) #flag>1为调暗,小于1为调亮
	return exposure.adjust_gamma(img, flag)

# 旋转
def rotate_img_bbox(img, point_chain, bboxes, angle=5, scale=1.):
	'''
	参考:https://blog.csdn.net/u014540717/article/details/53301195crop_rate
	输入:
		img:图像array,(h,w,c)
		bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
		angle:旋转角度
		scale:默认1
	输出:
		rot_img:旋转后的图像array
		rot_bboxes:旋转后的boundingbox坐标list
	'''
	#---------------------- 旋转图像 ----------------------
	w = img.shape[1]
	h = img.shape[0]
	# 角度变弧度
	rangle = np.deg2rad(angle)  # angle in radians
	# now calculate new image width and height
	nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
	nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
	# ask OpenCV for the rotation matrix
	rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
	# calculate the move from the old center to the new center combined
	# with the rotation
	rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5,0]))
	# the move only affects the translation, so update the translation
	# part of the transform
	rot_mat[0,2] += rot_move[0]
	rot_mat[1,2] += rot_move[1]
	# 仿射变换
	rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
	#---------------------- 矫正bbox坐标 ----------------------
	# rot_mat是最终的旋转矩阵
	# 获取原始bbox的四个中点，然后将这四个点转换到旋转后的坐标系下
	rot_bboxes = list()
	for bbox in bboxes:
		xmin = bbox[0]
		ymin = bbox[1]
		xmax = bbox[2]
		ymax = bbox[3]
		point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
		point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
		point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
		point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
		# 合并np.array
		concat = np.vstack((point1, point2, point3, point4))
		# 改变array类型
		concat = concat.astype(np.int32)
		# 得到旋转后的坐标
		rx, ry, rw, rh = cv2.boundingRect(concat)
		rx_min = rx
		ry_min = ry
		rx_max = rx+rw
		ry_max = ry+rh
		# 加入list中
		rot_bboxes.append([rx_min, ry_min, rx_max, ry_max, bbox[-1]])
	
	for tmp in point_chain["shapes"]:
		if tmp["label"] == "bullet":
			tmp["points"] = utils.mat_product(rot_mat, tmp["points"])

	return rot_img, point_chain, rot_bboxes

# 裁剪
def crop_img_bboxes(img, point_chain, bboxes):
	'''
	裁剪后的图片要包含所有的框
	输入:
		img:图像array
		bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
	输出:
		crop_img:裁剪后的图像array
		crop_bboxes:裁剪后的bounding box的坐标list
	'''
	#---------------------- 裁剪图像 ----------------------
	w = img.shape[1]
	h = img.shape[0]
	x_min = w   #裁剪后的包含所有目标框的最小的框
	x_max = 0
	y_min = h
	y_max = 0
	for bbox in bboxes:
		x_min = min(x_min, bbox[0])
		y_min = min(y_min, bbox[1])
		x_max = max(x_max, bbox[2])
		y_max = max(y_max, bbox[3])
	d_to_left = x_min           #包含所有目标框的最小框到左边的距离
	d_to_right = w - x_max      #包含所有目标框的最小框到右边的距离
	d_to_top = y_min            #包含所有目标框的最小框到顶端的距离
	d_to_bottom = h - y_max     #包含所有目标框的最小框到底部的距离
	#随机扩展这个最小框
	crop_x_min = int(x_min - random.uniform(0, d_to_left))
	crop_y_min = int(y_min - random.uniform(0, d_to_top))
	crop_x_max = int(x_max + random.uniform(0, d_to_right))
	crop_y_max = int(y_max + random.uniform(0, d_to_bottom))
	# 随机扩展这个最小框 , 防止别裁的太小
	# crop_x_min = int(x_min - random.uniform(d_to_left//2, d_to_left))
	# crop_y_min = int(y_min - random.uniform(d_to_top//2, d_to_top))
	# crop_x_max = int(x_max + random.uniform(d_to_right//2, d_to_right))
	# crop_y_max = int(y_max + random.uniform(d_to_bottom//2, d_to_bottom))
	#确保不要越界
	crop_x_min = max(0, crop_x_min)
	crop_y_min = max(0, crop_y_min)
	crop_x_max = min(w, crop_x_max)
	crop_y_max = min(h, crop_y_max)
	crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]
	#---------------------- 裁剪boundingbox ----------------------
	#裁剪后的boundingbox坐标计算
	crop_bboxes = list()
	for bbox in bboxes:
		crop_bboxes.append([bbox[0]-crop_x_min, bbox[1]-crop_y_min, bbox[2]-crop_x_min, bbox[3]-crop_y_min, bbox[-1]])
	
	for tmp in point_chain["shapes"]:
		if tmp["label"] == "bullet":
			tmp["points"] = utils.mat_minus([crop_x_min, crop_y_min], tmp["points"])

	return crop_img, point_chain, crop_bboxes

# 平移
def shift_pic_bboxes(img, point_chain, bboxes):
	'''
	参考:https://blog.csdn.net/sty945/article/details/79387054
	平移后的图片要包含所有的框
	输入:
		img:图像array
		bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
	输出:
		shift_img:平移后的图像array
		shift_bboxes:平移后的bounding box的坐标list
	'''
	#---------------------- 平移图像 ----------------------
	w = img.shape[1]
	h = img.shape[0]
	x_min = w   #裁剪后的包含所有目标框的最小的框
	x_max = 0
	y_min = h
	y_max = 0
	for bbox in bboxes:
		x_min = min(x_min, bbox[0])
		y_min = min(y_min, bbox[1])
		x_max = max(x_max, bbox[2])
		y_max = max(y_max, bbox[3])
	d_to_left = x_min           #包含所有目标框的最大左移动距离
	d_to_right = w - x_max      #包含所有目标框的最大右移动距离
	d_to_top = y_min            #包含所有目标框的最大上移动距离
	d_to_bottom = h - y_max     #包含所有目标框的最大下移动距离
	x = random.uniform(-(d_to_left-1) / 3, (d_to_right-1) / 3)
	y = random.uniform(-(d_to_top-1) / 3, (d_to_bottom-1) / 3)
	M = np.float32([[1, 0, x], [0, 1, y]])  #x为向左或右移动的像素值,正为向右负为向左; y为向上或者向下移动的像素值,正为向下负为向上
	shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
	#---------------------- 平移boundingbox ----------------------
	shift_bboxes = list()
	for bbox in bboxes:
		shift_bboxes.append([bbox[0]+x, bbox[1]+y, bbox[2]+x, bbox[3]+y, bbox[-1]])

	for tmp in point_chain["shapes"]:
		if tmp["label"] == "bullet":
			tmp["points"] = utils.mat_minus([-x, -y], tmp["points"])
	
	return shift_img, point_chain, shift_bboxes

# 镜像
def filp_pic_bboxes(img, point_chain, bboxes):
	'''
		参考:https://blog.csdn.net/jningwei/article/details/78753607
		平移后的图片要包含所有的框
		输入:
			img:图像array
			bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
		输出:
			flip_img:平移后的图像array
			flip_bboxes:平移后的bounding box的坐标list
	'''
	# ---------------------- 翻转图像 ----------------------
	flip_img = copy.deepcopy(img)
	if random.random() < 0.5:    #0.5的概率水平翻转，0.5的概率垂直翻转
		horizon = True
	else:
		horizon = False
	h,w,_ = img.shape
	if horizon: #水平翻转
		flip_img =  cv2.flip(flip_img, 1)
	else:
		flip_img = cv2.flip(flip_img, 0)
	# ---------------------- 调整boundingbox ----------------------
	flip_bboxes = list()
	for bbox in bboxes:
		x_min = bbox[0]
		y_min = bbox[1]
		x_max = bbox[2]
		y_max = bbox[3]
		if horizon:
			flip_bboxes.append([w-x_max, y_min, w-x_min, y_max, bbox[-1]])
		else:
			flip_bboxes.append([x_min, h-y_max, x_max, h-y_min, bbox[-1]])
	
	for tmp in point_chain["shapes"]:
		if tmp["label"] == "bullet":
			if horizon:
				tmp["points"] = utils.flip(w, h, horizon, tmp["points"])
			else:
				tmp["points"] = utils.flip(w, h, horizon, tmp["points"])

	return flip_img, point_chain, flip_bboxes

def gaussian_noise(image):
	variance = round(random.uniform(0.001,0.005),4)
	image_noise = skimage.util.random_noise(image, mode='gaussian', seed=None, clip=True, var = variance)
	return image_noise

def gaussian_filter(image):
	sigma = round(random.uniform(0.1,1),3)
	image_filter = filters.gaussian(image, sigma)
	return image_filter

def add_noise(image,image_name, index, output_image_path, xml_path, output_xml_path):
	if index == 1:
		'''salt noise'''
		variable = round(random.uniform(0.01,0.05),4)
		image_noise = skimage.util.random_noise(image, mode='salt', seed=None, clip=True, amount=variable)
	elif index == 2:
		'''pepper noise'''
		variable = round(random.uniform(0.005,0.01),4)
		image_noise = skimage.util.random_noise(image, mode='pepper', seed=None, clip=True, amount=variable)
	elif index == 3:
		'''salt + pepper noise'''
		variable = round(random.uniform(0.01,0.03),4)
		image_noise = skimage.util.random_noise(image, mode='s&p', seed=None, clip=True, amount=variable)
	elif index == 4:
		'''poisson noise'''
		image_noise = skimage.util.random_noise(image, mode='poisson', seed=None, clip=True)

	return image_noise

def add_filter(image, image_name, index, output_image_path, xml_path, output_xml_path):
	if index == 1:
		'''median filter'''
		image_filter = image
		sigma = round(random.uniform(0.3,0.6),3)
		for i in range(3):
			image_filter[:,:,i] = filters.median(image[:,:,i], disk(sigma))
	elif index == 2:
		'''minimum filter'''
		image_filter = image
		sigma = round(random.uniform(0.3,0.6),3)
		for i in range(3):
			image_filter[:,:,i] = sfr.minimum(image[:,:,i],disk(sigma))
	if exposure.is_low_contrast(image_filter) == False:
		return image_filter

def enh_gamma(image):
	'''lighter'''
	image = img_as_float(image)
	gam_factor = round(random.uniform(0.6,2),1)
	gam = exposure.adjust_gamma(image,gam_factor)
	return gam

def hist_unif(image):
	'''uniform histgram'''
	image_hist = exposure.equalize_hist(image)
	return image_hist

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
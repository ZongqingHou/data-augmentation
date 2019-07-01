import os
import numpy as np
import cv2

def calculate_mean():
	ims_path='/home/extension/datasets/bullect_collection/backup/tmp/images/'# 图像数据集的路径
	ims_list=os.listdir(ims_path)
	R_means=[]
	G_means=[]
	B_means=[]
	for im_list in ims_list:
		im=cv2.imread(ims_path+im_list)
	#extrect value of diffient channel
		im_R=im[:,:,0]
		im_G=im[:,:,1]
		im_B=im[:,:,2]
	#count mean for every channel
		im_R_mean=np.mean(im_R)
		im_G_mean=np.mean(im_G)
		im_B_mean=np.mean(im_B)
	#save single mean value to a set of means
		R_means.append(im_R_mean)
		G_means.append(im_G_mean)
		B_means.append(im_B_mean)
		print('图片：{} 的 BGR平均值为 \n[{}，{}，{}]'.format(im_list,im_R_mean,im_G_mean,im_B_mean) )
	#three sets  into a large set
	a=[R_means,G_means,B_means]
	mean=[0,0,0]
	#count the sum of different channel means
	mean[0]=np.mean(a[0])
	mean[1]=np.mean(a[1])
	mean[2]=np.mean(a[2])
	print('数据集的BGR平均值为\n[{}，{}，{}]'.format( mean[0],mean[1],mean[2]) )
	#cv.imread()读取Img时候将rgb转换为了bgr，谢谢taylover-pei的修正。

def minus_mean():
	import glob

	tmp_files = glob.glob("/home/extension/datasets/bullect_collection/backup/trainning/4/pairs/images/*.jpg")

	for tmp in tmp_files:
		img = cv2.imread(tmp)
		img[:,:,0] = img[:,:,0] - 124.1202406126534
		img[:,:,1] = img[:,:,1] - 127.84392627084473
		img[:,:,2] = img[:,:,2] - 115.4632036397791

		name = tmp.split('/')[-1].split('.')[0]
		cv2.imwrite('/home/extension/datasets/bullect_collection/backup/trainning/4/pairs/mean/' + name + '.jpg', img)

if __name__ == "__main__":
	# calculate_mean()
	minus_mean()
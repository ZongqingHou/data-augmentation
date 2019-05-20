import os
import skimage
import random
from skimage import data_dir, io, filters, img_as_float, exposure
from skimage.morphology import disk
import skimage.filters.rank as sfr
from PIL import Image, ImageEnhance
import xml.dom.minidom

def gaussion(image,image_name, output_image_path, xml_path, output_xml_path):
	image = io.imread(image)
	name = os.path.splitext(image_name)[0]
	try:
		'''gaussian noise'''
		variance = round(random.uniform(0.001,0.005),4)
		image_noise = skimage.util.random_noise(image, mode='gaussian', seed=None, clip=True, var = variance)
		io.imsave(output_image_path+name+'_gaunoise.jpg', image_noise)

		dom1 = xml.dom.minidom.parse(xml_path + name +'.xml')
		xmlfile1 = open(output_xml_path+name+'_gaunoise.xml','w')
		xmlfile1.write(dom1.toprettyxml(encoding='utf-8'))
		'''gaussian filter'''
		sigma = round(random.uniform(0.1,1),3)
		image_filter = filters.gaussian(image, sigma)
		io.imsave(output_image_path+name+'_gaufilter.jpg', image_filter)

		dom2 = xml.dom.minidom.parse(xml_path + name +'.xml')
		xmlfile2 = open(output_xml_path+name+'_gaufilter.xml','w')
		xmlfile2.write(dom2.toprettyxml(encoding='utf-8'))
	except ValueError:
		print('cannot save')

def add_noise(image,image_name, index, output_image_path, xml_path, output_xml_path):

	image = io.imread(image)
	name = os.path.splitext(image_name)[0]

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

	io.imsave(output_image_path+name+'_noise.jpg', image_noise)
	dom = xml.dom.minidom.parse(xml_path + name +'.xml')
	xmlfile = open(output_xml_path+name+'_noise.xml','w')
	xmlfile.write(dom.toprettyxml(encoding='utf-8'))

def add_filter(image, image_name, index, output_image_path, xml_path, output_xml_path):
	image = io.imread(image)
	name = os.path.splitext(image_name)[0]
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
		io.imsave(output_image_path+name+'_filter.jpg', image_filter)
		dom = xml.dom.minidom.parse(xml_path + name +'.xml')
		xmlfile = open(output_xml_path+name+'_filter.xml','w')
		xmlfile.write(dom.toprettyxml(encoding='utf-8'))

def enh_gamma(image,image_name, output_image_path, xml_path, output_xml_path):
	'''lighter'''
	image = io.imread(image)
	name = os.path.splitext(image_name)[0]
	image = img_as_float(image)
	gam_factor = round(random.uniform(0.6,2),1)
	gam = exposure.adjust_gamma(image,gam_factor)
	io.imsave(output_image_path+name+'_gam.jpg', gam)
	dom = xml.dom.minidom.parse(xml_path + name +'.xml')
	xmlfile = open(output_xml_path+name+'_gam.xml','w')
	xmlfile.write(dom.toprettyxml(encoding='utf-8'))

def hist_unif(image, image_name, output_image_path, xml_path, output_xml_path):
	'''uniform histgram'''
	image = io.imread(image)
	name = os.path.splitext(image_name)[0]
	image_hist = exposure.equalize_hist(image)
	io.imsave(output_image_path+ name +'_hist.jpg', image_hist)
	dom = xml.dom.minidom.parse(xml_path + name +'.xml')
	xmlfile = open(output_xml_path+name+'_hist.xml','w')
	xmlfile.write(dom.toprettyxml(encoding='utf-8'))

image_path = "/home/hdd/Desktop/dataset/yt20190404/"
output_image_path = "/home/hdd/Desktop/dataset/tmp_img/"
xml_path = "/home/hdd/Desktop/dataset/bt20190405/"
output_xml_path = "/home/hdd/Desktop/dataset/tmp_ano/"

filelist = os.listdir(image_path)
for files in filelist:
	print(files)
	image = image_path + files
	temp_noise = random.randint(1,4)
	temp_filter = random.randint(1,2)
	gaussion(image, files, output_image_path, xml_path, output_xml_path)
	add_noise(image, files, temp_noise, output_image_path, xml_path, output_xml_path)
	add_filter(image, files, temp_filter, output_image_path, xml_path, output_xml_path)
	enh_gamma(image, files, output_image_path, xml_path, output_xml_path)
	hist_unif(image, files, output_image_path, xml_path, output_xml_path)

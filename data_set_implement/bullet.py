import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

import argparse

parser = argparse.ArgumentParser(description='Label')
parser.add_argument('--ano_path', type=str, default="/home/extension/datasets/bullect_collection/backup/trainning/7/pairs/xml")
parser.add_argument('--img_path', type=str, default="/home/extension/datasets/bullect_collection/backup/trainning/7/pairs/images")
parser.add_argument('--label_path', type=str, default="/home/extension/datasets/bullect_collection/backup/trainning/7/pairs")
parser.add_argument('--id_list', type=str, default="/home/extension/datasets/bullect_collection/backup/trainning/7/image.txt")
parser.add_argument('--img_path_list', type=str, default="/home/extension/datasets/bullect_collection/backup/trainning/7")

args = parser.parse_args()
classes = ["bullet", "boundary", "target"]

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(image_id):
    global args

    in_file = open(args.ano_path + '/%s.xml'%(image_id))
    out_file = open(args.label_path + '/labels/%s.txt'%(image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

if not os.path.exists(args.label_path + '/labels/'):
    os.makedirs(args.label_path + '/labels/')
image_ids = open(args.id_list).read().strip().split()
list_file = open(args.img_path_list + '/bullet.txt', 'w')
for image_id in image_ids:
    list_file.write(args.img_path + '/%s.jpg\n'%(image_id))
    convert_annotation(image_id)
list_file.close()

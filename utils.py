# -*- coding=utf-8 -*-
import json
import glob

import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as DOC

# 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(xml_path):
    '''
    输入：
        xml_path: xml的文件路径
    输出：
        从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    '''
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(box[0].text)
        y_min = int(box[1].text)
        x_max = int(box[2].text)
        y_max = int(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords

#将bounding box信息写入xml文件中, bouding box格式为[[x_min, y_min, x_max, y_max, name]]
def generate_xml(img_name,coords,img_size, output_path):
    '''
    输入：
        img_name：图片名称，如a.jpg
        coords:坐标list，格式为[[x_min, y_min, x_max, y_max, name]]，name为概况的标注
        img_size：图像的大小,格式为[h,w,c]
        out_root_path: xml文件输出的根路径
    '''
    doc = DOC.Document()  # 创建DOM文档对象

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    title = doc.createElement('folder')
    title_text = doc.createTextNode('Tianchi')
    title.appendChild(title_text)
    annotation.appendChild(title)

    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)

    source = doc.createElement('source')
    annotation.appendChild(source)

    title = doc.createElement('database')
    title_text = doc.createTextNode('The Tianchi Database')
    title.appendChild(title_text)
    source.appendChild(title)

    title = doc.createElement('annotation')
    title_text = doc.createTextNode('Tianchi')
    title.appendChild(title_text)
    source.appendChild(title)

    size = doc.createElement('size')
    annotation.appendChild(size)

    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)

    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)

    for coord in coords:

        object = doc.createElement('object')
        annotation.appendChild(object)

        title = doc.createElement('name')
        title_text = doc.createTextNode(coord[4])
        title.appendChild(title_text)
        object.appendChild(title)

        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        object.appendChild(pose)
        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        object.appendChild(truncated)
        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        object.appendChild(difficult)

        bndbox = doc.createElement('bndbox')
        object.appendChild(bndbox)
        title = doc.createElement('xmin')
        title_text = doc.createTextNode(str(int(float(coord[0]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymin')
        title_text = doc.createTextNode(str(int(float(coord[1]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('xmax')
        title_text = doc.createTextNode(str(int(float(coord[2]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)
        title = doc.createElement('ymax')
        title_text = doc.createTextNode(str(int(float(coord[3]))))
        title.appendChild(title_text)
        bndbox.appendChild(title)

    # 将DOM对象doc写入文件
    f = open(output_path,'w')
    f.write(doc.toprettyxml(indent = ''))
    f.close()


def load_json(path='./aug_config.json'):
    with open(path, 'r') as file_buffer:
        load_dict = json.load(file_buffer)

    return load_dict

def save_json(load_dict, path='./aug_config.json'):
    with open(path, 'w') as file_buffer:
        json.dump(load_dict, file_buffer)

def read_files(img_path):
    img_path = img_path if img_path[-1] == '/' else img_path + '/'
    img_path += '*'

    img_list = glob.glob(img_path)
    return img_list

def max_min(points):
    x_collections = []
    y_collections = []

    for tmp_coord in points:
        x_collections.append(tmp_coord[0])
        y_collections.append(tmp_coord[1])

    return [min(x_collections), min(y_collections), max(x_collections), max(y_collections)]

def point_offset(points, offset):
    for tmp_coord in points:
        tmp_coord[0] += offset[0]
        tmp_coord[1] += offset[1]

    return points

def copy_img(img, src, dest):
    for index, tmp_coord in enumerate(src):
        img[dest[1]:dest[3], dest[0]:dest[2]] = img[src[1]:src[3], src[0]:src[2]]
        # img[dest[index][1]][dest[index][0]] = img[tmp_coord[1]][tmp_coord[0]]
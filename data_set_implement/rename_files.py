import os
import glob
import argparse

parser = argparse.ArgumentParser(description="data collection -> rename")
parser.add_argument("--src_img", type=str)
parser.add_argument("--src_json", type=str)
parser.add_argument("--src_xml", type=str)
parser.add_argument("--dest_img", type=str)
parser.add_argument("--dest_json", type=str)
parser.add_argument("--dest_xml", type=str)
parser.add_argument("--prefix", type=str, default="")
parser.add_argument("--start_index", type=int)

opt = parser.parse_args()

src_img_root_path = opt.src_img + "/" if opt.src_img[-1] != "/" else opt.src_img
src_json_root_path = opt.src_json + "/" if opt.src_json[-1] != "/" else opt.src_json
src_xml_root_path = opt.src_xml + "/" if opt.src_xml[-1] != "/" else opt.src_xml

dest_img_root_path = opt.dest_img + "/" if opt.dest_img[-1] != "/" else opt.dest_img
dest_json_root_path = opt.dest_json + "/" if opt.dest_json[-1] != "/" else opt.dest_json
dest_xml_root_path = opt.dest_xml + "/" if opt.dest_xml[-1] != "/" else opt.dest_xml

img_file_collection = glob.glob("{}*.jpg".format(src_img_root_path))

file_prefix = opt.prefix
file_rename = opt.start_index

if not os.path.exists(dest_img_root_path):
    os.makedirs(dest_img_root_path)

if not os.path.exists(dest_json_root_path):
    os.makedirs(dest_json_root_path)

if not os.path.exists(dest_xml_root_path):
    os.makedirs(dest_xml_root_path)

for tmp_file in img_file_collection:
    tmp_label = tmp_file.split("/")[-1].split(".")[0]
    os.rename(tmp_file, os.path.join(dest_img_root_path, file_prefix + str(file_rename) + ".jpg"))
    os.rename(os.path.join(src_json_root_path, tmp_label + ".json"), os.path.join(dest_json_root_path, file_prefix + str(file_rename) + ".json"))
    # os.rename(os.path.join(src_xml_root_path, tmp_label + ".xml"), os.path.join(dest_xml_root_path, file_prefix + str(file_rename) + ".xml"))

    file_rename += 1
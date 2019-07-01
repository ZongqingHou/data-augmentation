import glob
import argparse

parser = argparse.ArgumentParser(description='create id')
parser.add_argument('--img_folder', type=str)
parser.add_argument('--text_path', type=str)

args = parser.parse_args()

img_folder_path = args.img_folder + '/' if args.img_folder[-1] != '/' else args.img_folder

tt = glob.glob(img_folder_path + '*.jpg')
td = [int(tmp.split('/')[-1].split('.')[0]) for tmp in tt]
td.sort()

with open(args.text_path, 'w') as file:
	for tmp in td:
		file.write(str(tmp) + '\n')

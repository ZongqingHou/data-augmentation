import json

if __name__ == "__main__":
    import glob
    import argparse

    parse = argparse.ArgumentParser("coco style")
    parse.add_argument("--src_path", type=string)
    parse.add_argument("--dest_path", type=string)

    args = parse.parse_args()

    src_json_path = args.src_path if args.src_path[-1] == "/" else args.src_path + "/"
    json_files = glob.glob(src_json_path + "*.json")

    for tmp_file in json_files:
        with open(path, 'r', encoding="unicode_escape") as file_buffer:
		    load_dict = json.load(file_buffer)
        
        
    
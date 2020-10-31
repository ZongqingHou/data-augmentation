import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def dhash(data_buffer, output_buffer, delete_flag, min_distance):
    previous_id = None
    hash_list = []
    for tmp_line in data_buffer:
        tmp_message, tmp_id = tmp_line.split(" ")

        img = cv2.imread(tmp_message, cv2.IMREAD_GRAYSCALE)

        data = Image.fromarray(img)
        data_entry = np.array(data)
        valid_data_index = np.where(data_entry != 0)
        left_boundary, right_boundary = np.min(valid_data_index[1]), np.max(valid_data_index[1])

        face_width = abs(right_boundary - left_boundary)
        _, height = data.size

        depth_img = transforms.CenterCrop((height, face_width * 0.45))(data)
        target_size = int(face_width * 0.45 * 128 / 128)
        img_data = np.array(depth_img)

        top_bd = int(height / 2 - target_size * 0.65)
        top_bd = top_bd if top_bd > 0 else 0

        nose_region = img_data[top_bd: top_bd + target_size]
        nose_region = Image.fromarray(nose_region)
        img = np.array(nose_region)

        img = cv2.resize(img, (16, 16))
        tmp_img = np.where(img != 0)
        avg_np = np.mean(tmp_img)
        img = np.where(img > avg_np, 1, 0)

        if previous_id is None:
            print(tmp_id)
            previous_id = tmp_id
            hash_list.append(img)
            output_buffer.write(tmp_line)
        else:
            if tmp_id == previous_id:
                for tmp in hash_list:
                    flag = True
                    dis = np.bitwise_xor(tmp, img)

                    if np.sum(dis) < min_distance:
                        flag = False
                        break

                if flag:
                    hash_list.append(img)
                    output_buffer.write(tmp_line)
                else:
                    if delete_flag:
                        os.remove(tmp_message)
            else:
                print(tmp_id)
                previous_id = tmp_id
                hash_list = [img]
                output_buffer.write(tmp_line)



if __name__ == "__main__":
    import argparse

    parse = argparse.ArgumentParser(description="dhash the given datalist")
    parse.add_argument("--src_datalist", type=str)
    parse.add_argument("--output_datalist", type=str)
    parse.add_argument("--delete_flag", default=False, type=bool)
    parse.add_argument("--min_distance", default=10, type=int)

    opt = parse.parse_args()

    with open(opt.src_datalist, "r") as data_buffer_:
        with open(opt.output_datalist, "w") as output_buffer_:
            dhash(data_buffer_, output_buffer_, opt.delete_flag, opt.min_distance)

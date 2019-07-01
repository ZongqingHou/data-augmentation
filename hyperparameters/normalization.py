import cv2
import glob
import numpy as np

np.set_printoptions(threshold=np.inf)

def normalization(matrix):
    return matrix / np.linalg.norm(matrix, ord=2)

if __name__ == "__main__":
    img = cv2.imread("/home/extension/datasets/bullect_collection/src/total/images/chen_168_7.jpg")

    print(img.shape)
    img_ = np.zeros(img.shape)
    print(img_.shape)

    img_[:,:,0] = img[:,:,0] / np.linalg.norm(img[:,:,0], ord=2)
    img_[:,:,1] = img[:,:,1] / np.linalg.norm(img[:,:,1], ord=2)
    img_[:,:,2] = img[:,:,2] / np.linalg.norm(img[:,:,2], ord=2)

    # print(img.shape)
    # img_ = np.zeros(img.shape)
    # print(img_.shape)
    # cv2.normalize(src=img, dst=img_,norm_type=cv2.NORM_L2) 

    print(img_[:,:,0])
    cv2.imshow("tt", img_)
    cv2.waitKey(0)
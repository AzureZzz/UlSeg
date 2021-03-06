import pandas as pd
import os
import numpy as np
from skimage.measure import regionprops
from skimage.measure import label as sklabel
from PIL import Image



def save_csv(path):
    img_path = os.path.join(path, 'image')
    csv_path = os.path.join(path, 'origin.csv')
    img_list = os.listdir(img_path)
    category = [x[1] for x in img_list]
    np_data = np.array([np.array(img_list), np.array(category)]).T
    save = pd.DataFrame(np_data, columns=['ID', 'CATE'])
    save.to_csv(csv_path, index=False)


def rename(path):
    img_path = os.path.join(path, 'image')
    mask_path = os.path.join(path, 'mask')
    csv_path = os.path.join(path, 'origin.csv')
    img_list = os.listdir(img_path)
    category = [x[1] for x in img_list]
    for i in range(len(img_list)):
        old_img = os.path.join(img_path, img_list[i])
        old_mask = os.path.join(mask_path, img_list[i])
        new_name = str(i) + '.png'
        new_img = os.path.join(img_path, new_name)
        new_mask = os.path.join(mask_path, new_name)
        os.rename(old_img, new_img)
        os.rename(old_mask, new_mask)
        img_list[i] = new_name
    np_data = np.array([np.array(img_list), np.array(category)]).T
    save = pd.DataFrame(np_data, columns=['ID', 'CATE'])
    save.to_csv(csv_path, index=False)

def largestConnectComponent(bw_img):
    if np.sum(bw_img) == 0:
        return bw_img
    labeled_img, num = sklabel(bw_img, neighbors=4, background=0, return_num=True)
    if num == 1:
        return bw_img

    max_label = 0
    max_num = 0
    for i in range(0, num):
        print(i)
        if np.sum(labeled_img == (i + 1)) > max_num:
            max_num = np.sum(labeled_img == (i + 1))
            max_label = i + 1
    mcr = (labeled_img == max_label)
    return mcr.astype(np.int)

def preprocess(mask_c1_array_biggest, c1_size=256):
    if np.sum(mask_c1_array_biggest) == 0:
        minr, minc, maxr, maxc = [0, 0, c1_size, c1_size]
    else:
        region = regionprops(mask_c1_array_biggest)[0]
        minr, minc, maxr, maxc = region.bbox

    dim1_center, dim2_center = [(maxr + minr) // 2, (maxc + minc) // 2]
    max_length = max(maxr - minr, maxc - minc)

    max_lengthl = int((c1_size / 256) * 80)
    preprocess1 = int((c1_size / 256) * 19)
    pp22 = int((c1_size / 256) * 31)

    if max_length > max_lengthl:
        ex_pixel = preprocess1 + max_length // 2
    else:
        ex_pixel = pp22 + max_length // 2

    dim1_cut_min = dim1_center - ex_pixel
    dim1_cut_max = dim1_center + ex_pixel
    dim2_cut_min = dim2_center - ex_pixel
    dim2_cut_max = dim2_center + ex_pixel

    if dim1_cut_min < 0:
        dim1_cut_min = 0
    if dim2_cut_min < 0:
        dim2_cut_min = 0
    if dim1_cut_max > c1_size:
        dim1_cut_max = c1_size
    if dim2_cut_max > c1_size:
        dim2_cut_max = c1_size
    return [dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max]


def RGBA2RGB(path):
    img_list = os.listdir(path)
    for img in img_list:
        img_path = os.path.join(path,img)
        image = Image.open(img_path).convert('RGB')
        image.save(img_path)


if __name__ == '__main__':
    # path = r'dataset/our_large_old/origin'
    # save_csv(path)
    # path = r'dataset/our_large_old/preprocessed/stage1'
    # rename(path)
    RGBA2RGB('dataset/our_large/origin/image3')




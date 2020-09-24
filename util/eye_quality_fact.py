import numpy as np
import cv2


def ellipse2circle(param):
    if len(param) == 5:
        r = int((param[2] + param[3]) / 2)
        new_param = (param[0], param[1], r)
    elif len(param) == 3:
        new_param = param
    else:
        new_param = ()
    return new_param


def points_in_circle(mask, circle):
    points = []
    radius = circle[2]**2
    for y, x in np.where(mask == True):
        if (x - circle[0])**2 + (y - circle[1])**2 <= radius:
            points.append((x, y))
    return set(points)


def points_out_circle(mask, circle):
    points = []
    radius = circle[2]**2
    for y, x in np.where(mask == True):
        if (x - circle[0])**2 + (y - circle[1])**2 > radius:
            points.append((x, y))
    return set(points)


def points_between_circle(mask, circle1, circle2):
    points = []
    radius1 = circle1[2]**2
    radius2 = circle2[2]**2
    for x in range(mask.shape[1]):
        for y in range(mask.shape[0]):
            conditions = (mask[y, x] == True) and (
                (x - circle1[0])**2 + (y - circle1[1])**2 > radius1) and (
                    (x - circle2[0])**2 + (y - circle2[1])**2 <= radius2)
            if conditions:
                points.append((x, y))
    return set(points)


# #############################################################################


def sharpness(img):
    # Sharpness (defocus/motion)
    gaussianX = cv2.Sobel(img, cv2.CV_16U, 1, 0)
    gaussianY = cv2.Sobel(img, cv2.CV_16U, 1, 0)
    fm = np.mean(np.sqrt(gaussianX**2 + gaussianY**2))
    return fm


def iris_size(iris_param):
    # Iris size (iris radius in pixel)
    if len(iris_param) == 3:
        r = iris_param[2]
    elif len(iris_param) == 5:
        r = int((iris_param[2] + iris_param[3]) / 2)
    else:
        r = 0
    return r


def dilation(iris_param, pupil_param):
    # Pupil iris ratio (ratio of pupil diameter over iris diameter)
    if len(iris_param) == 3 and len(pupil_param) == 3:
        ri = iris_param[2]
        pi = pupil_param[2]
    elif len(iris_param) == 5 and len(pupil_param) == 5:
        ri = int((iris_param[2] + iris_param[3]) / 2)
        pi = int((pupil_param[2] + pupil_param[3]) / 2)
    else:
        ri, pi = 0, 0

    if ri != 0 and pi != 0:
        dilation_ratio = pi / ri * 100
    else:
        dilation_ratio = -1
    return dilation_ratio


def gray_level_spread(img, mask):
    # Gray level spread
    img = img.astype(np.int)
    img[mask != True] = -1
    usable_pix_num = np.sum(mask)
    ent = 0.0
    for i in range(256):
        p = np.sum(img == i) / usable_pix_num
        if p != 0:
            ent -= p * np.log2(p)
    return ent


def usable_area(mask, iris_param, pupil_param):
    # Usable iris area (percentage of usable iris area)

    # usable_area = points_between_circle(mask, pupil_param, iris_param)
    # all_area = points_between_circle(
    #     np.ones_like(mask).astype(np.bool), pupil_param, iris_param)
    # usable_area_ratio = len(usable_area) / len(all_area) * 100
    # return usable_area_ratio

    usable_area = mask.sum()
    all_area = np.pi * (iris_param[2]**2 - pupil_param[2]**2)
    usable_area_ratio = usable_area / all_area * 100
    return usable_area_ratio


def iris_sclera_contrast(img, mask, iris_param, pupil_param):
    pass


def iris_pupil_contrast(img, mask, iris_param, pupil_param):
    pass


# ############################################################################
def ini_reader(filepath):
    with open(filepath, 'r') as f:
        data = [x.strip() for x in f.readlines()]
    if len(data) == 15:
        # ellipse
        iris_param = [float(x.split('=')[1]) for x in data[2:7]]
        pupil_param = [float(x.split('=')[1]) for x in data[10:15]]
        flag = 'ellipse'
    elif len(data) == 11:
        # circle
        iris_param = [float(x.split('=')[1]) for x in data[2:5]]
        pupil_param = [float(x.split('=')[1]) for x in data[8:11]]
        flag = 'circle'
    else:
        # None
        iris_param = None
        pupil_param = None
        flag = None
    return flag, iris_param, pupil_param


if __name__ == "__main__":
    import os
    from glob import glob
    from tqdm import tqdm
    import shutil

    with open('data/cx2/train.txt', 'r') as f:
        namelist = [x.split('.')[0] for x in f.readlines()]

    quality_map = []
    errors = []
    for filename in tqdm(namelist, ncols=79, ascii=True):
        try:
            img = cv2.imread(
                'data/cx2/Image/{}.bmp'.format(filename), 0)
            mask = cv2.imread(
                'data/cx2/Result/Mask/{}.png'.format(filename), 0)
            if mask.shape != img.shape:
                mask = cv2.resize(mask,
                                  (img.shape[1], img.shape[0])).astype(np.bool)
            else:
                mask = mask.astype(np.bool)
            flag, iris_param, pupil_param = ini_reader(
                'data/cx2/Result/seg_param/{}.ini'.format(filename))

            fm = sharpness(img)
            ir = iris_size(iris_param)
            dr = dilation(iris_param, pupil_param)
            gls = gray_level_spread(img, mask)
            uar = usable_area(mask, iris_param, pupil_param)
            quality_map.append((filename + '.bmp', fm, ir, dr, gls, uar))
        except Exception as e:
            print(e)

    with open('cx2_train_quality.txt', 'w') as f:
        for line in quality_map:
            f.write('{}, {} {} {} {} {}\n'.format(*line))


    with open('data/cx1/train.txt', 'r') as f:
        namelist = [x.split('.')[0] for x in f.readlines()]

    quality_map = []
    errors = []
    for filename in tqdm(namelist, ncols=79, ascii=True):
        try:
            img = cv2.imread(
                'data/cx1/Image/{}.bmp'.format(filename), 0)
            mask = cv2.imread(
                'data/cx1/Mask/{}.png'.format(filename), 0)
            if mask.shape != img.shape:
                mask = cv2.resize(mask,
                                  (img.shape[1], img.shape[0])).astype(np.bool)
            else:
                mask = mask.astype(np.bool)
            flag, iris_param, pupil_param = ini_reader(
                'data/cx1/seg_param/{}.ini'.format(filename))

            fm = sharpness(img)
            ir = iris_size(iris_param)
            dr = dilation(iris_param, pupil_param)
            gls = gray_level_spread(img, mask)
            uar = usable_area(mask, iris_param, pupil_param)
            quality_map.append((filename + '.bmp', fm, ir, dr, gls, uar))
        except Exception as e:
            print(e)

    with open('cx1_train_quality.txt', 'w') as f:
        for line in quality_map:
            f.write('{}, {} {} {} {} {}\n'.format(*line))
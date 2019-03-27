import os
import sys

import cv2 as cv
import numpy as np
from tqdm import tqdm
import pandas as pd
import time

sys.path.append(os.getcwd())
# from utils.prepare.utils import orderConvex, shrink_poly

from utils import orderConvex, shrink_poly

DATA_FOLDER = "data/demo/source/"
OUTPUT = "data/dataset/mlt/"
MAX_LEN = 1200
MIN_LEN = 600

label = pd.read_csv(os.path.join(DATA_FOLDER, "label.csv"))

im_fns = os.listdir(DATA_FOLDER)
im_fns.sort()

if not os.path.exists(os.path.join(OUTPUT, "image")):
    os.makedirs(os.path.join(OUTPUT, "image"))
if not os.path.exists(os.path.join(OUTPUT, "label")):
    os.makedirs(os.path.join(OUTPUT, "label"))

for im_fn in tqdm(im_fns):
    try:
        _, fn = os.path.split(im_fn)
        bfn, ext = os.path.splitext(fn)
        if ext.lower() not in ['.jpg', '.png']:
            continue

        img_path = os.path.join(DATA_FOLDER,im_fn)
        img = cv.imread(img_path)
        img_size = img.shape
        # 旋转竖的图片
        (h, w) = img_size[:2]  # 10
        if h < w:
            print('pass')
        if h > w:
            image = cv.transpose(img)
            image = cv.flip(image, 0)

            img_size = image.shape
            im_size_min = np.min(img_size[0:2])
            im_size_max = np.max(img_size[0:2])

            im_scale = float(600) / float(im_size_min)
            if np.round(im_scale * im_size_max) > 1200:
                im_scale = float(1200) / float(im_size_max)
            new_w = int(img_size[0] * im_scale)
            new_h = int(img_size[1] * im_scale)

            new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
            new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

            re_im = cv.resize(image, (new_h, new_w), interpolation=cv.INTER_LINEAR)
            re_size = re_im.shape

            polys = []
            lines = label.loc[label['FileName'] == im_fn]
            for index, row in lines.iterrows():
                # x1, y1, x2, y2, x3, y3, x4, y4 = row['x1'], row['y1'], row['x2'], row['y2'], row['x3'], row['y3'], row[
                #     'x4'], row['y4']
                x1, y1, x2, y2, x3, y3, x4, y4 = row['y1'], w - row['x1'], row['y2'], w - row['x2'], row['y3'], w - row[
                    'x3'], row['y4'], w - row['x4']

                # for line in lines:
                #     splitted_line = line.strip().lower().split(',')
                #     x1, y1, x2, y2, x3, y3, x4, y4 = map(float, splitted_line[:8])
                poly = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape([4, 2])
                poly[:, 0] = poly[:, 0] / img_size[1] * re_size[1]
                poly[:, 1] = poly[:, 1] / img_size[0] * re_size[0]
                poly = orderConvex(poly)
                polys.append(poly)

                # cv.polylines(re_im, [poly.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)

            res_polys = []
            for poly in polys:
                # delete polys with width less than 10 pixel
                if np.linalg.norm(poly[0] - poly[1]) < 10 or np.linalg.norm(poly[3] - poly[0]) < 10:
                    continue
                res = shrink_poly(poly)

                # for p in res:
                #     cv.polylines(re_im, [p.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=1)

                res = res.reshape([-1, 4, 2])
                for r in res:
                    x_min = np.min(r[:, 0])
                    y_min = np.min(r[:, 1])
                    x_max = np.max(r[:, 0])
                    y_max = np.max(r[:, 1])

                    res_polys.append([x_min, y_min, x_max, y_max])

            cv.imwrite(os.path.join(OUTPUT, "image", fn), re_im)
            with open(os.path.join(OUTPUT, "label", bfn) + ".txt", "w") as f:
                for p in res_polys:
                    line = ",".join(str(p[i]) for i in range(4))
                    f.writelines(line + "\r\n")

    except:
        print("Error processing {}".format(im_fn))

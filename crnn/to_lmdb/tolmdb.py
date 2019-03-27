import os
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import re
from PIL import Image
import numpy as np
import imghdr
import pandas as pd
import os


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.fromstring(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
        imgH, imgW = img.shape[0], img.shape[1]
    except:
        return False
    else:
        if imgH * imgW == 0:
            return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            if (isinstance(v, bytes)):
                txn.put(k.encode(), v)
            else:
                txn.put(k.encode(), v.encode())


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert (len(imagePathList) == len(labelList))
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = os.path.join('/Users/liyangyang/Downloads/df/hanzi/traindataset', 'validation', imagePathList[i][0])
        # print(imagePath)
        label = ''.join(labelList[i])
        # print(label)
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue

        [x1, y1, x2, y2, x3, y3, x4, y4] = imagePathList[i][1:9]
        # x1, y1, x2, y2, x3, y3, x4, y4 = int(x1), int(y1), int(x2), int(y2), int(x3), int(y3), int(x4), int(y4)
        xmin = min(x1, x2, x3, x4)
        xmax = max(x1, x2, x3, x4)
        ymin = min(y1, y2, y3, y4)
        ymax = max(y1, y2, y3, y4)
        image = cv2.imread(imagePath)

        h1, w1, c1 = image.shape
        # print(image.shape)

        if h1 > w1:
            image = image[ymin:ymax, xmin:xmax]
            image = cv2.transpose(image)
            image = cv2.flip(image, 0)
            # print(image.shape)

            # imageBin = cv2.imencode(image, cv2.IMREAD_GRAYSCALE)

            img_encode = cv2.imencode('.jpg', image)[1]
            data_encode = np.array(img_encode)
            imageBin = data_encode.tostring()

            if checkValid:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label
            if lexiconList:
                lexiconKey = 'lexicon-%09d' % cnt
                cache[lexiconKey] = ' '.join(lexiconList[i])
            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
            cnt += 1
            # print(cnt)
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


if __name__ == '__main__':
    DATA_FOLDER = "/Users/liyangyang/Downloads/df/hanzi/traindataset/"

    outputPath = "./all"
    imageList = []

    label = pd.read_csv(DATA_FOLDER + 'verify_lable.csv')
    # label2 = pd.read_csv(DATA_FOLDER + 'train_lable.csv')
    labelList = []
    for index, row in label.iterrows():
        imageList.append([row['FileName'], row['x1'], row['y1'], row['x2'], row['y2'], row['x3'], row['y3'], row['x4'],
                          row['y4']])
        labelList.append(row['text'])

    createDataset(outputPath, imageList, labelList)

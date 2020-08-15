# Принцип работы:
#
# 1. В папку in помещаются картинки
# 2. Запускается настоящий скрипт
# 3. В папке out появляется файл output.csv, представляющий из себя таблицу со следующими полями:
#     > n
#       номер наблюдения
#     > image_name
#       имя картинки, с которой производилось распознавание
#     > recognized
#       распознанные номера авто

import os
import numpy as np
import pandas
import sys
import matplotlib.image as mpimg

from worker import Worker

NOMEROFF_NET_DIR = os.path.abspath('./')

MASK_RCNN_DIR = os.path.join(NOMEROFF_NET_DIR, 'Mask_RCNN')
MASK_RCNN_LOG_DIR = os.path.join(NOMEROFF_NET_DIR, 'logs')


bork = Worker(NOMEROFF_NET_DIR, MASK_RCNN_DIR, MASK_RCNN_LOG_DIR,
              load_model="latest", options_detector="latest", text_detector_module="eu", load_text_detector="latest")


# dira = './in/'
if len(sys.argv) != 4:
    raise Exception("You must provide a target address and a destination address and output file name (without extention)")
dira = sys.argv[1]
ims = []
detected = []
for file in os.listdir(dira):
    detecty = bork.detect(os.path.join(dira, file))
    detected.append(detecty)
    ims.append([file] * len(detecty))


detected_ = [y for x in detected for y in x]
ims_ = [y for x in ims for y in x]
n_ = list(range(len(ims_)))

data = pandas.DataFrame(data={'n': n_, 'image_name': ims_, 'recognized': detected_})

# data.to_csv('./out/output.csv', index=False)
data.to_csv(os.path.join(sys.argv[2], '{}.csv'.format(sys.argv[3])), index=False)

#
import os
import numpy as np
import pandas
import sys
import matplotlib.image as mpimg


#
from NomeroffNet import filters, RectDetector, TextDetector, OptionsDetector, Detector, textPostprocessing, textPostprocessingAsync


#


class Worker:
    def __init__(self, NOMEROFF_NET_DIR, MASK_RCNN_DIR, MASK_RCNN_LOG_DIR, load_model="latest", options_detector="latest",
                 text_detector_module="eu", load_text_detector="latest"):
        sys.path.append(NOMEROFF_NET_DIR)
        # Initialize npdetector with default configuration file.
        self.nnet = Detector(MASK_RCNN_DIR, MASK_RCNN_LOG_DIR)
        self.nnet.loadModel(load_model)

        self.rectDetector = RectDetector()

        self.optionsDetector = OptionsDetector()
        self.optionsDetector.load(options_detector)

        # Initialize text detector.
        self.textDetector = TextDetector.get_static_module(text_detector_module)()
        self.textDetector.load(load_text_detector)

    def detect(self, img_path):
        # Detect numberplate
        img = mpimg.imread(img_path)
        NP = self.nnet.detect([img])

        # Generate image mask.
        cv_img_masks = filters.cv_img_mask(NP)

        # Detect points.
        arrPoints = self.rectDetector.detect(cv_img_masks)
        zones = self.rectDetector.get_cv_zonesBGR(img, arrPoints)

        # find standart
        regionIds, stateIds, countLines = self.optionsDetector.predict(zones)
        regionNames = self.optionsDetector.getRegionLabels(regionIds)

        # find text with postprocessing by standart
        textArr = self.textDetector.predict(zones)
        textArr = textPostprocessing(textArr, regionNames)

        return textArr

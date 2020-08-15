#
import numpy
import pandas
import subprocess
from PIL import Image


#


#
def catch_the_car(bork, detector, custom, input_image_path, output_detected_image_path, output_cropped_image_path, comparator_env, probability_threshold=30, prev=None):

    # Stage 1: Detect the car

    detections = detector.detectCustomObjectsFromImage(custom_objects=custom, input_image=input_image_path, output_image_path=output_detected_image_path, minimum_percentage_probability=probability_threshold)

    names = []
    probabilities = []
    box_points = []
    box_areas = []

    for eachObject in detections:
        names.append(eachObject["name"])
        probabilities.append(eachObject["percentage_probability"])
        box_points.append(eachObject["box_points"])
        area = (box_points[-1][2] - box_points[-1][0]) * (box_points[-1][3] - box_points[-1][1])
        box_areas.append(area)

    max_ix = box_areas.index(max(box_areas))

    # Stage 2: Crop the car

    img = Image.open(input_image_path)
    cropped_img = img.crop(box_points[max_ix])
    cropped_img.save(output_cropped_image_path)

    # Stage 3: Detect the numberplate

    numberplate_code = bork.detect(output_cropped_image_path)[0]

    # Stage 4: Compare with previous

    if prev is None:

        cmp_score_im, cmp_score_numberplate_code = numpy.nan, numpy.nan

    else:

        prev_im_path = prev['im']
        prev_im_numberplate_code = prev['nc']
        im_comparator = prev['im_c']
        numberplate_code_comparator = prev['nc_c']

        target_script = './compare.py'
        subprocess.run([comparator_env,
                        target_script, im_comparator, numberplate_code_comparator,
                        prev_im_path, output_cropped_image_path,
                        prev_im_numberplate_code, numberplate_code])

        result_frame = pandas.read_csv('./result.csv')
        cmp_score_im, cmp_score_numberplate_code = result_frame.values[0, :].tolist()

    return cropped_img, numberplate_code, cmp_score_im, cmp_score_numberplate_code

#
import os
from imageai.Detection import ObjectDetection


# simply from the guide
model_path = './models'
image_path = './image_similary'
detected_path = './image_detected'

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(model_path, "yolo.h5"))
detector.loadModel()

detections = detector.detectObjectsFromImage(input_image=os.path.join(image_path , "1_0.jpg"), output_image_path=os.path.join(detected_path , "1_0_detected.jpg"), minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")


# with vehicles only
model_path = './models'
image_path = './image_similary'
detected_path = './image_detected'

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(model_path, "yolo.h5"))
detector.loadModel()

custom = detector.CustomObjects(car=True, motorcycle=True, bus=True, truck=True)

detections = detector.detectCustomObjectsFromImage( custom_objects=custom, input_image=os.path.join(image_path , "6.jpg"), output_image_path=os.path.join(detected_path , "6_detected.jpg"), minimum_percentage_probability=30)

for eachObject in detections:
    print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )
    print("--------------------------------")


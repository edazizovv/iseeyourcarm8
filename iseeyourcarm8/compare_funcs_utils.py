#
import tensorflow


#


#
def load_img(path):
    img = tensorflow.io.read_file(path)
    img = tensorflow.io.decode_jpeg(img, channels=3)
    img = tensorflow.image.resize_with_pad(img, 224, 224)
    img = tensorflow.image.convert_image_dtype(img, tensorflow.float32)[tensorflow.newaxis, ...]
    return img

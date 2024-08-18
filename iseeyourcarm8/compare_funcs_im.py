#
import requests
import tensorflow_hub
from scipy import spatial


#
from compare_funcs_utils import load_img


#
def deepai_im_cmp(im0, im1):
    k = 'C:/Users/MainUser/Desktop/deepai_APIkey.txt'
    crs = open(k, "r")
    for columns in (raw.strip().split() for raw in crs):
        API_KEY = columns[0]
    r = requests.post(
        "https://api.deepai.org/api/image-similarity",
        files={
            'image1': open(im0, 'rb'),
            'image2': open(im1, 'rb'),
        },
        headers={'api-key': API_KEY}
    )
    return r.json()['output']['distance']


def tf_mobilenet_im_cmp(im0, im1):

    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    module = tensorflow_hub.load(module_handle)

    img0, img1 = load_img(im0), load_img(im1)
    features0, features1 = module(img0), module(img1)

    similarity = 1 - spatial.distance.cosine(features0, features1)
    rounded_similarity = int((similarity * 10000)) / 10000.0

    return rounded_similarity

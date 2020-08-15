#
import requests

# TEST I: Using API from https://deepai.org/machine-learning-model/image-similarity

# STATUS: WORKS

k = 'C:/Users/MainUser/Desktop/deepai_APIkey.txt'
crs = open(k, "r")
for columns in (raw.strip().split() for raw in crs):
    API_KEY = columns[0]

r = requests.post(
    "https://api.deepai.org/api/image-similarity",
    files={
        'image1': open('./image_similary/1_0.jpg', 'rb'),
        'image2': open('./image_similary/1_1.jpg', 'rb'),
    },
    headers={'api-key': API_KEY}
)
print(r.json())

r = requests.post(
    "https://api.deepai.org/api/image-similarity",
    files={
        'image1': open('./image_similary/1_0.jpg', 'rb'),
        'image2': open('./image_similary/3_0.jpg', 'rb'),
    },
    headers={'api-key': API_KEY}
)
print(r.json())

r = requests.post(
    "https://api.deepai.org/api/image-similarity",
    files={
        'image1': open('./image_similary/3_0.jpg', 'rb'),
        'image2': open('./image_similary/3_2.jpg', 'rb'),
    },
    headers={'api-key': API_KEY}
)
print(r.json())

# TEST II: Using a Guide https://towardsdatascience.com/image-similarity-detection-in-action-with-tensorflow-2-0-b8d9a78b2509

# STATUS: WORKS

# get_image_feature_vectors.py#################################################
# Imports and function definitions
#################################################
# For running inference on the TF-Hub module with Tensorflow
import tensorflow as tf
import tensorflow_hub as hub  # For saving 'feature vectors' into a txt file
import numpy as np  # Glob for reading file names in a folder
import glob
import os.path


##################################################################################################
# This function:
# Loads the JPEG image at the given path
# Decodes the JPEG image to a uint8 W X H X 3 tensor
# Resizes the image to 224 x 224 x 3 tensor
# Returns the pre processed image as 224 x 224 x 3 tensor
#################################################
def load_img(path):  # Reads the image file and returns data type of string
    img = tf.io.read_file(path)  # Decodes the image to W x H x 3 shape tensor with type of uint8
    img = tf.io.decode_jpeg(img, channels=3)  # Resizes the image to 224 x 224 x 3 shape tensor
    img = tf.image.resize_with_pad(img, 224, 224)  # Converts the data type of uint8 to float32 by adding a new axis
    # img becomes 1 x 224 x 224 x 3 tensor with data type of float32
    # This is required for the mobilenet model we are using
    img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]

    return img  #################################################
    # This function:
    # Loads the mobilenet model in TF.HUB
    # Makes an inference for all images stored in a local folder
    # Saves each of the feature vectors in a file
    #################################################


def get_image_feature_vectors(from_dir, to_dir):
    # Definition of module with using tfhub.dev
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    # Loads the module
    module = hub.load(module_handle)  # Loops through all images in a local folder

    ## feature_bucket = []
    for filename in glob.glob('{0}/*.jpg'.format(from_dir)):
        print(filename)  # Loads and pre-process the image
        img = load_img(filename)  # Calculate the image feature vector of the img
        features = module(img)  # Remove single-dimensional entries from the 'features' array
        feature_set = np.squeeze(features)

        # Saves the image feature vectors into a file for later use
        outfile_name = os.path.basename(filename) + ".npz"

        out_path = os.path.join(to_dir, outfile_name)  # Saves the 'feature_set' to a text file
        np.savetxt(out_path, feature_set, delimiter=',')
        ## feature_bucked.append(feature_set)
    ## return feature_bucket

from_dir = './image_similary'
to_dir = './feature_similary'
get_image_feature_vectors(from_dir, to_dir)


# Numpy for loading image feature vectors from file
import numpy as np

# Time for measuring the process time
import time

# Glob for reading file names in a folder
import glob
import os.path

# json for storing data in json file
import json

# Annoy and Scipy for similarity calculation
from annoy import AnnoyIndex
from scipy import spatial

def cluster(to_dir):

    start_time = time.time()

    print("---------------------------------")
    print ("Step.1 - ANNOY index generation - Started at %s" %time.ctime())
    print("---------------------------------")

    # Defining data structures as empty dict
    file_index_to_file_name = {}
    file_index_to_file_vector = {}

    # Configuring annoy parameters
    dims = 1792
    n_nearest_neighbors = 20
    trees = 10000

    # Reads all file names which stores feature vectors
    allfiles = glob.glob('{0}/*.npz'.format(to_dir))

    t = AnnoyIndex(dims, metric='angular')

    for file_index, i in enumerate(allfiles):

        # Reads feature vectors and assigns them into the file_vector
        file_vector = np.loadtxt(i)

        # Assigns file_name, feature_vectors and corresponding product_id
        file_name = os.path.basename(i).split('.')[0]
        file_index_to_file_name[file_index] = file_name
        file_index_to_file_vector[file_index] = file_vector

        # Adds image feature vectors into annoy index
        t.add_item(file_index, file_vector)

        print("---------------------------------")
        print("Annoy index     : %s" %file_index)
        print("Image file name : %s" %file_name)
        print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))


    # Builds annoy index
    t.build(trees)

    print ("Step.1 - ANNOY index generation - Finished")
    print ("Step.2 - Similarity score calculation - Started ")

    named_nearest_neighbors = []

    # Loops through all indexed items
    for i in file_index_to_file_name.keys():

        # Assigns master file_name, image feature vectors and product id values
        master_file_name = file_index_to_file_name[i]
        master_vector = file_index_to_file_vector[i]

        # Calculates the nearest neighbors of the master item
        nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)

        # Loops through the nearest neighbors of the master item
        for j in nearest_neighbors:

            print(j)

            # Assigns file_name, image feature vectors and product id values of the similar item
            neighbor_file_name = file_index_to_file_name[j]
            neighbor_file_vector = file_index_to_file_vector[j]

            # Calculates the similarity score of the similar item
            similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
            rounded_similarity = int((similarity * 10000)) / 10000.0

            # Appends master product id with the similarity score
            # and the product id of the similar items
            named_nearest_neighbors.append({
            'similarity': rounded_similarity,
            'master_pi': master_file_name,
            'similar_pi': neighbor_file_name})

    print("---------------------------------")
    print("Similarity index       : %s" %i)
    print("Master Image file name : %s" %file_index_to_file_name[i])
    print("Nearest Neighbors.     : %s" %nearest_neighbors)
    print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))


    print ("Step.2 - Similarity score calculation - Finished ")

    # Writes the 'named_nearest_neighbors' to a json file
    with open('nearest_neighbors.json', 'w') as out:
        json.dump(named_nearest_neighbors, out)

    print ("Step.3 - Data stored in 'nearest_neighbors.json' file ")
    print("--- Prosess completed in %.2f minutes ---------" % ((time.time() - start_time)/60))


#
from PIL import Image

img = './image_detected/1_0_detected.jpg'
croppy = [46, 113, 695, 442]
img = Image.open(img)
cropped_img = img.crop(croppy)
cropped_img.show()
cropped_img.save('./image_chopped/1_0_chop.jpg')

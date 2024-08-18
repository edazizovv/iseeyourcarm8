#
import sys
import pandas
from pyxdameraulevenshtein import normalized_damerau_levenshtein_distance


#
from compare_funcs_im import deepai_im_cmp, tf_mobilenet_im_cmp


#
cmp_im, cmp_code = sys.argv[1], sys.argv[2]
im0, im1 = sys.argv[3], sys.argv[4]
code0, code1 = sys.argv[5], sys.argv[6]

if cmp_im == 'deepai':
    cmp_im_value = deepai_im_cmp(im0, im1)
elif cmp_im == 'tf_mobilenet':
    cmp_im_value = tf_mobilenet_im_cmp(im0, im1)
else:
    cmp_im_value = 'UNKNOWN'

if cmp_code == 'damerau_levenshtein':
    cmp_code_value = normalized_damerau_levenshtein_distance(code0, code1)
else:
    cmp_code_value = 'UNKNOWN'

result_frame = pandas.DataFrame(data={'cmp_im': [cmp_im_value], 'cmp_code': [cmp_code_value]})
result_frame.to_csv('./result.csv', index=False)

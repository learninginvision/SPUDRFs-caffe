import sys,os
import numpy as np
sys.path.insert(0, './caffe_soft/python')

import caffe
from caffe import layers as L, params as P, to_proto
import time
from PIL import Image

def Predict(para_dict):
    test_txt = para_dict['test']
    predict_txt = para_dict['predict'] 
    deploy_file = para_dict['deploy'] 
    model_file = para_dict['model'] 
    root_dir=  './Morph_mtcnn_1.3_0.35_0.3/' 
    TEST_TIMES = 1

    net = caffe.Net(deploy_file,model_file,caffe.TEST)
    caffe.set_mode_gpu()

    decoding_time = 0.0
    diff_sum = 0.0
    count = 0

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.array([112,112,112]))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2,1,0))

    f = open(test_txt)
    f_save = open(predict_txt,'w')
    for eachline in f:
        img_name,true_age = eachline.strip().split(' ')
        true_age = float(true_age)
        full_path = root_dir + img_name
        predict_age_t = 0.0

        for iter_time in range(0,TEST_TIMES):
            image = caffe.io.load_image(full_path)
            net.blobs['data'].data[...] = transformer.preprocess('data', image)

            output = net.forward()

            start_time = time.clock()
            predict_age_t += output['pred']
            end_time = time.clock()
            decoding_time += (end_time - start_time)

        predict_age = predict_age_t/float(TEST_TIMES)

        diff = abs(true_age - predict_age)
        diff_sum += diff

        print('Image name: %s, Truth age: %s, Predicted age: %f, absoulte diff: %f' % (img_name, true_age, predict_age, diff))
        print >> f_save, 'Image name: %s, Truth age: %s, Predicted age: %f, absoulte diff: %f' % (img_name, true_age, predict_age, diff)

        count += 1

    print(count)
    print(diff_sum/count)
    f.close()
    f_save.close()
    return diff_sum/count

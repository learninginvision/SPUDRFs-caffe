import sys,os
import numpy as np
# import cv2
# sys.path.insert(0, '/root/data/aishijie/Project/entropy/caffe-DRFs/python')
# sys.path.insert(0, '/root/data/meng/DRFs1/caffe-drf/python')
# sys.path.insert(0, '/root/data/meng/caffe_entropy3/python')
sys.path.insert(0, '/root/data/meng/caffe_soft/python')
import caffe
from caffe import layers as L, params as P, to_proto
import time
import PIL
from PIL import Image


def Predict(para_dict):
    test_txt = para_dict['test'] # './images/epfl/trainLeft3-0.15.txt'  # test_data_color
    predict_txt = para_dict['predict'] # './MAE/EPFL/MAEOnTrainLeft3-0.15.txt'
    deploy_file = para_dict['deploy'] # './tmp/EPFL/EPFL-deploy.prototxt'
    model_file = para_dict['model'] # '/hdd2/meng/model/3EPFL-VGG-NO_iter_20000.caffemodel'
    root_dir=  '/root/data/aishijie/Project/Morph_mtcnn_1.3_0.35_0.3/'  # "/home/aishijie/meng/images/epfl/tripod-seq/"  #

    TEST_TIMES = 1

    net = caffe.Net(deploy_file,model_file,caffe.TEST)
    caffe.set_mode_gpu()
    # caffe.set_device(1)

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

            # obtain the probabilities of classifiers
            output = net.forward()

            # Local Age Decoding
            start_time = time.clock()


            predict_age_t += output['pred']
            end_time = time.clock()
            decoding_time += (end_time - start_time)
        # -------------------------------------------
        predict_age = predict_age_t/float(TEST_TIMES)

        diff = abs(true_age - predict_age)
        if diff > 25:
            diff = 36 - diff
        diff_sum += diff


        print('Image name: %s, Truth age: %s, Predicted age: %f, absoulte diff: %f' % (img_name, true_age, predict_age, diff))
        print >> f_save, 'Image name: %s, Truth age: %s, Predicted age: %f, absoulte diff: %f' % (img_name, true_age, predict_age, diff)

        count += 1
    print(count)
    print(diff_sum/count)
    f.close()
    f_save.close()
    return diff_sum/count

    # with open(predict_txt, 'r') as f:
    #     lines = f.readlines()
    # new_lines = []
    # for line in lines:
    #     keep, diff = line.split('diff: ')
    #     diff = float(diff)
    #     if diff > 30:
    #         diff = 36 - diff
    #     new_line = keep + 'diff: ' + str(diff) + '\n'
    #     new_lines.append(new_line) 
    # with open(predict_txt, 'w') as f:
    #     f.writelines(new_lines)


if __name__ == '__main__':
    phase = 'train'
    if phase == 'train':
        exp = '1'
        pace = '0'
        para_dict = {}
        para_dict['predict'] = './MAE/mae' + exp + '/MAEOnTrainLeft' +pace+ '-0.15.txt'
        para_dict['test'] = './images/txt-' + exp + '/trainLeft' +pace+ '-0.15.txt'

        para_dict['model'] = './checkpoints/M' + exp + '/0EPFL-VGG-NO_iter_40000.caffemodel'
        para_dict['deploy'] = './tmp/Exp'+ exp + '/' +pace+ 'EPFL-VGG-NO-deploy.prototxt'

        diff = Predict(para_dict)
    else:
        diff_ave = []
        for pace in range(5,6):
            print('Pace : %d' % pace)
            exp = '24'
            tmp_dir = './testMAE/exp' + exp + '/'
            if not os.path.isdir(tmp_dir):
                os.makedirs(tmp_dir)
            para_dict = {}
            para_dict['predict'] = './testMAE/exp'+ exp + '/'+ exp + '-' + str(pace) + 'PaceMAEOnTest.txt' 
            para_dict['model'] = './checkpoints/M'+ exp +'/' + str(pace) + 'EPFL-VGG-NO_iter_10000.caffemodel'

            para_dict['test'] = './images/epfl/EPFL-testB.txt' 
            para_dict['deploy'] = './tmp/EPFL/Exp'+ exp + '/' + str(pace) + 'EPFL-VGG-NO-deploy.prototxt'
            
            diff = Predict(para_dict)
            with open('./Entropy.txt', 'r') as f:
                lines = f.readlines()
            fn = tmp_dir + str(pace) + 'entropy.txt'
            with open(fn, 'w') as f:
                f.writelines(lines)
            diff = np.squeeze(diff)
            # print(diff)
            line = 'pace {}: {} \n'.format(pace, diff)
            diff_ave.append(line)
        for line in diff_ave:
            print(line)
        
        fn = './testMAE/exp' + exp + '/'+ exp + '-paceMAE.txt'
        with open(fn, 'w') as f:
            f.writelines(diff_ave) 

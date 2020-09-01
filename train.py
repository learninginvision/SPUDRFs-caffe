# begin at 19:53
import sys, os, re, urllib, time
time.sleep(0)
sys.path.insert(0, './caffe_soft/python')
import caffe
from caffe import layers as L, params as P, to_proto
from caffe.coord_map import crop
from caffe.proto import caffe_pb2
from os.path import join, splitext, abspath, exists, dirname, isdir, isfile
from datetime import datetime
from scipy.io import savemat
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse

class SPUDRFs():
    def __init__(self, parser_dict):

        self.parser_dict = parser_dict
        self.record_filename = self.parser_dict['record']   
        self.data = self.parser_dict['data']  
        self.save = self.parser_dict['save']
        self.checkdir(self.save)
        self.traintxt = self.parser_dict['traintxt']
        self.testtxt = self.parser_dict['testtxt']
        self.base_weights = self.parser_dict['base_weights']
        self.tmp_dir = self.parser_dict['tmp_dir']
        self.checkdir(self.tmp_dir)
        self.ntree = 5
        self.treeDepth = 6
        self.nout = 128
        self.drop = False
        self.init = 'init'
        self.gpu = 1
        self.cs = [i for i in range(11)]

        with open(self.traintxt,'r') as f:
            self.nTrain = len(f.readlines())
        with open(self.testtxt,'r') as f:
            self.nTest = len(f.readlines())

        self.testdir = "./Morph_mtcnn_1.3_0.35_0.3/"
        self.traindir = "./Morph_mtcnn_1.3_0.35_0.3/"

        if parser_dict['pace'] == 10:
            self.maxIter = 80000
        else:
            self.maxIter = 40000
        self.test_interval = 500  #
        self.test_batch_size = 16   #
        self.train_batch_size = 32  #
        self.test_iter = int(np.ceil(self.nTest / self.test_batch_size))
    
    def checkdir(self, tmp_dir):
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)

    def make_net(self, phase='train'):
        n = caffe.NetSpec()
        if phase == 'train':
            batch_size = self.train_batch_size
            n.data, n.label = L.ImageMultilabelData(ntop=2,image_multilabel_data_param=dict(source=self.traintxt,root_folder=self.traindir,\
            shuffle=True,batch_size=batch_size,new_height=256,new_width=256,label_dim=2),
            transform_param=dict(mean_value=112,crop_size=224, mirror=True)
            )
            n.label1, n.label2 = L.Slice(n.label,ntop=2,slice_param=dict(axis=1,slice_point=1),name='slice')
        elif phase == 'test':
            batch_size = self.test_batch_size
            n.data, n.label = L.ImageData(ntop=2,image_data_param=dict(source=self.testtxt,root_folder=self.testdir,\
            batch_size=batch_size,new_height=256,new_width=256),transform_param=dict(mean_value=112, crop_size=224, mirror=True) 
            )
        if phase == 'deploy':
            n.data = L.Input(shape=dict(dim=[1,3,224,224]))  

        n.conv1_1, n.relu1_1 = conv_relu(n.data, 64, mult=[10,10,20,0])
        n.conv1_2, n.relu1_2 = conv_relu(n.relu1_1, 64, mult=[10,10,20,0])
        n.pool1 = max_pool(n.relu1_2)

        n.conv2_1, n.relu2_1 = conv_relu(n.pool1, 128, mult=[10,1,20,0])
        n.conv2_2, n.relu2_2 = conv_relu(n.relu2_1, 128, mult=[10,1,20,0])
        n.pool2 = max_pool(n.relu2_2)

        n.conv3_1, n.relu3_1 = conv_relu(n.pool2, 256, mult=[10,1,20,0])
        n.conv3_2, n.relu3_2 = conv_relu(n.relu3_1, 256, mult=[10,1,20,0])
        n.conv3_3, n.relu3_3 = conv_relu(n.relu3_2, 256, mult=[10,1,20,0])
        n.pool3 = max_pool(n.relu3_3)

        n.conv4_1, n.relu4_1 = conv_relu(n.pool3, 512)
        n.conv4_2, n.relu4_2 = conv_relu(n.relu4_1, 512)
        n.conv4_3, n.relu4_3 = conv_relu(n.relu4_2, 512)
        n.pool4 = max_pool(n.relu4_3)
        
        n.conv5_1, n.relu5_1 = conv_relu(n.pool4, 512)
        n.conv5_2, n.relu5_2 = conv_relu(n.relu5_1, 512)
        n.conv5_3, n.relu5_3 = conv_relu(n.relu5_2, 512)
        n.pool5 = max_pool(n.relu5_3)

        n.fc6 = L.InnerProduct(n.pool5, num_output=4096, bias_term=True, weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0),
                    param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
        n.relu6 = L.ReLU(n.fc6, in_place=True)
        n.drop6 = L.Dropout(n.relu6, in_place=True, dropout_ratio=0.5)

        n.fc7 = L.InnerProduct(n.drop6, num_output=4096, bias_term=True, weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0),
                    param=[dict(lr_mult=10, decay_mult=1), dict(lr_mult=20, decay_mult=0)])
        n.relu7 = L.ReLU(n.fc7, in_place=True)
        n.drop7 = L.Dropout(n.relu7, in_place=True, dropout_ratio=0.5)

        if self.nout > 0:
            assert(self.nout >= int(pow(2, self.treeDepth - 1) - 1))
            nout = self.nout
        else:
            if self.ntree == 1:
                nout = int(pow(2, self.treeDepth - 1) - 1)
            else:
                nout = int((pow(2, self.treeDepth - 1) - 1) * self.ntree * 2 / 3)
        n.fc8 = L.InnerProduct(n.drop7, num_output=nout, bias_term=True, weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0),
                    param=[dict(lr_mult=1, decay_mult=1), dict(lr_mult=2, decay_mult=0)], name='fc8-101')#name='fc8a')

        if phase=='train':
            all_data_vec_length = int(50)
            n.loss = L.NeuralDecisionRegForestWithLoss(n.fc8, n.label1, n.label2, 
                param=[dict(lr_mult=0, decay_mult=0), dict(lr_mult=0, decay_mult=0),dict(lr_mult=0, decay_mult=0)], 
                neural_decision_forest_param=dict(depth=self.treeDepth, num_trees=self.ntree, num_classes=1, iter_times_class_label_distr=20, 
                iter_times_in_epoch=50, all_data_vec_length=all_data_vec_length, drop_out=self.drop, init_filename=self.init,record_filename=self.record_filename), 
                name='probloss1')
        
        elif phase=='test':
            n.pred = L.NeuralDecisionRegForest(n.fc8, n.label, neural_decision_forest_param=dict(depth=self.treeDepth, num_trees=self.ntree, num_classes=1), name='probloss1')
            n.MAE = L.MAE(n.pred, n.label)
            n.CS0 = L.CS(n.pred, n.label,cs_param = dict(lll = self.cs[0]))
            n.CS1 = L.CS(n.pred, n.label,cs_param = dict(lll = self.cs[1]))
            n.CS2 = L.CS(n.pred, n.label,cs_param = dict(lll = self.cs[2]))
            n.CS3 = L.CS(n.pred, n.label,cs_param = dict(lll = self.cs[3]))
            n.CS4 = L.CS(n.pred, n.label,cs_param = dict(lll = self.cs[4]))
            n.CS5 = L.CS(n.pred, n.label,cs_param = dict(lll = self.cs[5]))
            n.CS6 = L.CS(n.pred, n.label,cs_param = dict(lll = self.cs[6]))
            n.CS7 = L.CS(n.pred, n.label,cs_param = dict(lll = self.cs[7]))
            n.CS8 = L.CS(n.pred, n.label,cs_param = dict(lll = self.cs[8]))
            n.CS9 = L.CS(n.pred, n.label,cs_param = dict(lll = self.cs[9]))
            n.CS10 = L.CS(n.pred, n.label,cs_param = dict(lll = self.cs[10]))
        
        elif phase=='deploy':
            n.pred = L.NeuralDecisionRegForest(n.fc8, neural_decision_forest_param=dict(depth=self.treeDepth, num_trees=self.ntree, num_classes=1), name='probloss1')
        return n.to_proto()

    def make_solver(self):
        s = caffe_pb2.SolverParameter()
        s.type = 'SGD'
        s.display = 10
        s.base_lr = 0.2
        s.lr_policy = "step"
        s.gamma = 0.5
        s.momentum = 0.9
        s.stepsize = 10000
        s.max_iter = self.maxIter
        s.snapshot = 40000
        snapshot_prefix = join(dirname(__file__), self.save)
        if not isdir(snapshot_prefix):
            os.makedirs(snapshot_prefix)
        s.snapshot_prefix = join(snapshot_prefix, self.data)
        s.train_net = join(self.tmp_dir, self.data + '-train' + '.prototxt')
        s.test_net.append(join(self.tmp_dir, self.data + '-test' + '.prototxt'))
        s.test_interval = 10000 # will test mannualy
        s.test_iter.append(self.test_iter)
        s.test_initialization = True
        return s

    def train(self):
        
        with open(join(self.tmp_dir, self.data + '-train'  + '.prototxt'), 'w') as f:
            f.write(str(self.make_net()))
        with open(join(self.tmp_dir, self.data + '-test'  + '.prototxt'), 'w') as f:
            f.write(str(self.make_net('test')))
        with open(join(self.tmp_dir, self.data + '-deploy'  + '.prototxt'), 'w') as f:
            f.write(str(self.make_net('deploy')))
        with open(join(self.tmp_dir, self.data + '-solver' + '.prototxt'), 'w') as f:
            f.write(str(self.make_solver()))

        iter = 0
        mae, cs__0, cs__1, cs__2, cs__3, cs__4, cs__5, cs__6, cs__7, cs__8, cs__9, cs__10 = [], [], [], [], [], [], [], [], [], [], [], [] 
        
        caffe.set_mode_gpu()
        solver = caffe.SGDSolver(join(self.tmp_dir, self.data + '-solver' + '.prototxt'))

        base_weights = self.base_weights
        
        if not isfile(base_weights):
            print "There is not base model to %s"%(base_weights)
        
        solver.net.copy_from(base_weights)
        for p in solver.net.params:
            param = solver.net.params[p][0].data[...]
            print "  After layer \"%s\":, parameter[0] mean=%f, std=%f"%(p, param.mean(), param.std())

        iter = 0
        while iter < self.maxIter:
            
            solver.step(self.test_interval)
            
            solver.test_nets[0].share_with(solver.net)
            
            mae1 = np.float32(0.0)
            cs0 = np.float32(0.0)
            cs1 = np.float32(0.0)
            cs2 = np.float32(0.0)
            cs3 = np.float32(0.0)
            cs4 = np.float32(0.0)
            cs5 = np.float32(0.0)
            cs6 = np.float32(0.0)
            cs7 = np.float32(0.0)
            cs8 = np.float32(0.0)
            cs9 = np.float32(0.0)
            cs10 = np.float32(0.0)

            for t in range(self.test_iter):
                output= solver.test_nets[0].forward()
                mae1 += output['MAE']
                cs0 += output['CS0']
                cs1 += output['CS1']
                cs2 += output['CS2']
                cs3 += output['CS3']
                cs4 += output['CS4']
                cs5 += output['CS5']
                cs6 += output['CS6']
                cs7 += output['CS7']
                cs8 += output['CS8']
                cs9 += output['CS9']
                cs10 += output['CS10']

            mae1 /= self.test_iter
            cs0 /= self.test_iter
            cs1 /= self.test_iter
            cs2 /= self.test_iter
            cs3 /= self.test_iter
            cs4 /= self.test_iter
            cs5 /= self.test_iter
            cs6 /= self.test_iter
            cs7 /= self.test_iter
            cs8 /= self.test_iter
            cs9 /= self.test_iter
            cs10 /= self.test_iter

            mae.append(mae1)
            cs__0.append(cs0)
            cs__1.append(cs1)
            cs__2.append(cs2)
            cs__3.append(cs3)
            cs__4.append(cs4)
            cs__5.append(cs5)
            cs__6.append(cs6)
            cs__7.append(cs7)
            cs__8.append(cs8)
            cs__9.append(cs9)
            cs__10.append(cs10)


            iter = iter + self.test_interval
            # print args
            print "Iter%d, currentMAE=%.4f, bestMAE=%.4f, currentCS0=%.4f, bestCS0=%.4f, currentCS1=%.4f, \
            bestCS1=%.4f, currentCS2=%.4f, bestCS2=%.4f, currentCS3=%.4f, bestCS3=%.4f, currentCS4=%.4f, \
            bestCS4=%.4f, currentCS5=%.4f, bestCS5=%.4f, currentCS6=%.4f, bestCS6=%.4f, currentCS7=%.4f, \
            bestCS7=%.4f, currentCS8=%.4f, bestCS8=%.4f, currentCS9=%.4f, bestCS9=%.4f, currentCS10=%.4f, \
            bestCS10=%.4f"%(iter, mae[-1], min(mae), cs__0[-1], max(cs__0), cs__1[-1], max(cs__1), cs__2[-1], max(cs__2), cs__3[-1], max(cs__3), cs__4[-1], max(cs__4), cs__5[-1], max(cs__5), cs__6[-1], max(cs__6), cs__7[-1], max(cs__7), cs__8[-1], max(cs__8), cs__9[-1], max(cs__9), cs__10[-1], max(cs__10))

        mae = np.array(mae, dtype=np.float32)
        cs = np.array(cs__5, dtype=np.float32)
        sav_fn = join(self.tmp_dir, "MAE-%stree%ddepth%dtime%s"%(
                self.data, self.ntree, self.treeDepth, datetime.now().strftime("M%mD%d-H%HM%MS%S")))
        np.save(sav_fn+'.npy', mae)
        sav_fn_ = join(self.tmp_dir, 'MAE-%s' % self.data)
        np.save(sav_fn_ +'.npy', mae)
        mat_dict = dict({'mae':mae,'cs':cs})
        mat_dict.update(self.parser_dict)  # save args to .mat
        savemat(sav_fn+'.mat', mat_dict)

        print "Best MAE=%.4f, Best CS0=%.4f, Best CS1=%.4f, Best CS2=%.4f, Best CS3=%.4f, Best CS4=%.4f, Best CS5=%.4f, Best CS6=%.4f, Best CS7=%.4f, Best CS8=%.4f, Best CS9=%.4f, Best CS10=%.4f."%(
            mae.min(), max(cs__0), max(cs__1), max(cs__2), max(cs__3), max(cs__4), max(cs__5), max(cs__6), max(cs__7), max(cs__8), max(cs__9), max(cs__10))
        
        print "Done! Results saved at \'"+sav_fn+"\'"

def conv_relu(bottom, nout, ks=3, stride=1, pad=1, mult=[1,1,2,0]):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
            num_output=nout, pad=pad, 
            weight_filler=dict(type='gaussian', std=0.005),
            param=[dict(lr_mult=mult[0], decay_mult=mult[1]), dict(lr_mult=mult[2], decay_mult=mult[3])])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks=2, stride=2):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


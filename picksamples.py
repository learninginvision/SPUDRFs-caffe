import os
import numpy as np 
import predict
import random

class PickSamples():

    def __init__(self, exp=0, percent=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1], pace=0, alpha=[0.9, 0.9, 0.87, 0.85, 0.8, 0.8], 
                ent_threshold=-4.2, diff_threshold=1.5, ent_pick_per=0.1, random_pick=False, 
                soft=False,root='.', max_step=40000):
        self.max_step = max_step
        self.root = root
        self.exp = exp
        self.root_image = self.root + '/images/txt-' + str(self.exp) + '/'
        self.root_MAE = self.root + '/MAE/mae' + str(self.exp) + '/'
        self.root_entropy = self.root + '/entropy/E' + str(self.exp) + '/'
        self.checkdir(self.root_image)
        self.checkdir(self.root_MAE)
        self.checkdir(self.root_entropy)
        self.percent = percent
        self.pace = pace
        self.ent_threshold = ent_threshold
        self.diff_threshold = diff_threshold
        self.ent_pick_per = ent_pick_per
        self.prefix_pick = 'trainPick'
        self.prefix_left = 'trainLeft'
        self.prefix_MAE = 'MAEOnTrainLeft'
        self.prefix_pick_ent = 'pick_ent'
        self.prefix_ent = 'entropy'
        self.scale = 0.15
        self.alpha = alpha
        self.random_pick = random_pick
        self.soft = soft

        self.fn_traintxt0 = './images/MORPH-train.txt'
        train_images = self.readtxt(self.fn_traintxt0)
        self.pace_samples = [int(p*len(train_images)) for p in self.percent]
        print('percent: ', self.percent)
        print('pace_samples: ', self.pace_samples)

        imgs = []
        labels = []
        for t in train_images:
            img, label = t.split(' ')
            imgs.append(img)
            labels.append(float(label))
        self.dict_train = dict(zip(imgs, labels))
        # print(self.root_image, self.root_MAE, self.root_entropy)

    def checkdir(self, tmp_dir):
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)

    def readtxt(self, fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
        return lines

    def savetxt(self, fn, lines):
        with open(fn, 'w') as f:
            f.writelines(lines)

    def get_prefix(self, phase):
        if phase == 'pick' :
            prefix = self.prefix_pick
            root = self.root_image
        elif phase == 'left':
            prefix = self.prefix_left
            root = self.root_image
        elif phase == 'pick_ent':
            prefix = self.prefix_pick_ent
            root = self.root_image
        elif phase == 'MAE':
            prefix = self.prefix_MAE
            root = self.root_MAE
        elif phase == 'entropy':
            prefix = self.prefix_ent
            root = self.root + '/'
        else:
            raise NameError('illegal phase in get prefix')
            
        return (prefix, root)

    def get_fn(self, pace, phase='pick'):
        prefix, root = self.get_prefix(phase)
        if 'entropy' in prefix:
            fn = 'Entropy.txt'
            fn = root + fn
        else:
            fn = prefix + str(int(pace)) + '-' + str(self.scale) + '.txt'
            fn = os.path.join(root, fn)

        return fn

    def predict(self, pace=0, soft=False):
        para_dict = {}
        exp = str(self.exp)
        if soft:
            para_dict['predict'] = './MAE/mae' + exp + '/MAEOnTrainPick{}-0.15.txt'.format(pace)
            para_dict['test'] = './images/txt-' + exp + '/trainPick{}-0.15.txt'.format(pace)
            para_dict['model'] = './checkpoints/M'+exp+'/'+str(pace-1)+'EPFL-VGG-NO_iter_{}.caffemodel'.format(self.max_step)
            para_dict['deploy'] = './tmp/Exp' + exp + '/' +str(pace-1)+ 'EPFL-VGG-NO-deploy.prototxt'
        if not soft:
            para_dict['predict'] = './MAE/mae' + exp + '/{}MAEOnTrainLeft0-0.15.txt'.format(pace-1)
            para_dict['test'] = './images/txt-' + exp + '/trainLeft0-0.15.txt'
            para_dict['model'] = './checkpoints/M' + exp + '/' + str(pace-1) + 'EPFL-VGG-NO_iter_{}.caffemodel'.format(self.max_step)
            para_dict['deploy'] = './tmp/Exp' + exp + '/' +str(pace-1)+ 'EPFL-VGG-NO-deploy.prototxt'
        diff = predict.Predict(para_dict)
            
    def pick(self, pace=0):
        '''
        pace represent the txt need to be generated
        '''
        pick,left,pick_soft,left_soft,pred_sort,pick_ent = [],[],[],[],[],[]
        if pace == 0:
            pick = [] 
            left = self.readtxt(self.fn_traintxt0)
            
            if self.soft:
                for line in left:
                    img, label = line.strip('\n').split(' ')
                    left_soft.append(img + ' ' + label + ' ' + '10000' + '\n')
        else:
            fn_train_previous = self.get_fn(pace-1, phase='pick')
            fn_predictMAE = self.get_fn(pace-1, phase='MAE')
            fn_train_pick = self.get_fn(pace, phase='pick')
            fn_train_left = self.get_fn(pace, phase='left')
            print('fn_train_previous: %s, Length: %d' % (fn_train_previous, len(self.readtxt(fn_train_previous))))
            print('fn_predictMAE: %s, Length: %d' % (fn_predictMAE, len(self.readtxt(fn_predictMAE))))
            print('fn_train_pick: %s' % (fn_train_pick))
            print('fn_train_left: %s' % (fn_train_left))

            # print('fn_train_previous: %s, Length: %d' % (fn_train_previous, len(self.readtxt(fn_train_previous))))
            # print('fn_predictMAE: %s, Length: %d' % (fn_predictMAE, len(self.readtxt(fn_predictMAE))))
            # print('fn_train_pick: %s, Length: %d' % (fn_train_pick, len(self.readtxt(fn_train_pick))))
            # print('fn_train_left: %s, Length: %d' % (fn_train_left, len(self.readtxt(fn_train_left))))
            pred = self.readtxt(fn_predictMAE)
            
            # fn_entropy = self.get_fn(pace-1, phase='entropy')
            # print('fn_entropy: %s' % fn_entropy)
            # entropy = self.readtxt(fn_entropy)
            # fn_save_ent = self.root_entropy + '/entropy' + str(pace-1) + '.txt'
            # self.savetxt(fn_save_ent, entropy)
            # new
            fn_entropy = self.root_entropy + '/entropy' + str(pace-1) + '.txt'
            entropy = self.readtxt(fn_entropy)
            assert len(entropy) == len(pred), 'entropy do not equal to pred %d vs %d' % (len(entropy), len(pred))
            for i, p in enumerate(pred):
                diff = float(p.split(':')[-1])
                if diff > self.diff_threshold:
                    diff = self.diff_threshold
                # img = p.split(':')[1][1:22]
                img = p.split(':')[1].split(',')[0][1:]
                ent = float(entropy[i].split(':')[-1])
                if self.ent_threshold < 0:
                    if ent < self.ent_threshold:
                        ent = self.ent_threshold
                diff = diff - self.alpha[pace-1] * ent
                pred_sort.append((img, diff))

            pred_sort.sort(key=lambda x:x[1])
            print('pred_sort: %d' % len(pred_sort))
            for i in range(len(pred_sort)):
                p = pred_sort[i][0]
                # print(p)
                line = p + ' ' + str(self.dict_train[p]) + '\n'
                if i < self.pace_samples[pace-1]:
                    pick.append(line)
                else:
                    left.append(line)
            print('pace pick: %d' % len(pick))
            tem = self.readtxt(fn_train_previous)
            
            # add more 10% according entropy
            if self.ent_pick_per > 0:
                if self.random_pick:
                    lines = self.readtxt(self.fn_traintxt0)
                    random.shuffle(lines)
                    if self.ent_pick_per < 1:
                        len_lim = int(len(idx_ent) * self.ent_pick_per)
                    else:
                        len_lim = self.ent_pick_per
                    pick_ent = lines[:len_lim]
                else:
                    ent_all_txt_ = './entropy/E{}/entropyAll{}.txt'.format(self.exp, pace-1)
                    fn_predictMAE = './MAE/mae{}/{}MAEOnTrainLeft0-0.15.txt'.format(self.exp, pace-1)
                    entropy2, pred2 = [], []
                    if os.path.exists(ent_all_txt_) and os.path.exists(fn_predictMAE):
                        entropy2 = self.readtxt(ent_all_txt_)
                        pred2 = self.readtxt(fn_predictMAE)
                    else:
                        self.predict(pace=pace)
                        entropy2 = self.readtxt('./Entropy.txt')
                        pred2 = self.readtxt(fn_predictMAE)
                        self.savetxt(ent_all_txt_, entropy2)
                    assert len(entropy2) == len(pred2), 'entropy do not equal to pred %d vs %d' % (len(entropy), len(pred))
                    ent_sort = []
                    for e in entropy2:
                        ent = float(e.split(':')[-1])
                        ent_sort.append(ent)
                    ent_sort_np = np.array(ent_sort)
                    idx_ent = np.argsort(-ent_sort_np)
                    if self.ent_pick_per < 1:
                        len_lim = int(len(idx_ent) * self.ent_pick_per)
                    else:
                        len_lim = self.ent_pick_per
                    idx_ent_pick = idx_ent[:len_lim]
                    
                    for i in range(idx_ent_pick.shape[0]):
                        idx = idx_ent_pick[i]
                        p = pred2[idx]
                        # img = p.split(':')[1][1:22]
                        img = p.split(':')[1].split(',')[0][1:]
                        label = str(int(self.dict_train[img]))
                        line = img + ' ' + label + '\n'
                        pick_ent.append(line)
                
            l1, l2, l3 = len(pick), len(tem), len(pick_ent)
            pick = pick + tem + pick_ent
            print('pick{} = pace_pick{} + previous{} + ent_pick{}'.format(l1+l2+l3, l1,l2,l3))
        
        fn_save_pick = self.get_fn(pace, phase='pick')
        fn_save_left = self.get_fn(pace, phase='left')
        fn_save_pick_ent = self.get_fn(pace, phase='pick_ent')
        self.savetxt(fn_save_pick_ent, pick_ent)
        self.savetxt(fn_save_pick, pick)
        self.savetxt(fn_save_left, left)

        
        if self.soft and pace > 0:
    
            ent_all_txt_ = './entropy/E{}/entropyAll{}.txt'.format(self.exp, pace-1)
            fn_predictMAE = './MAE/mae{}/{}MAEOnTrainLeft0-0.15.txt'.format(self.exp, pace-1)
            ent_all = self.readtxt(ent_all_txt_)
            mae_all = self.readtxt(fn_predictMAE)
            print('mae_all: ', len(mae_all))
            assert len(ent_all) == len(mae_all), 'entropy do not equal to pred %d vs %d' % (len(ent_all), len(mae_all))
            mae_pick, ent_pick, img_all = [], [], []
            for mae in mae_all:
                img = mae.split(':')[1].split(',')[0][1:]
                img_all.append(img)
            for p in pick:
                img = p.split(' ')[0]
                idx = img_all.index(img)
                mae_pick.append(mae_all[idx])
                ent_pick.append(ent_all[idx]) 
            print('mae_pick: ', len(mae_pick))
            
            ###################
            # mae_pick_txt = './MAE/mae{}/MAEOnTrainPick{}-0.15.txt'.format(self.exp, pace)
            # ent_pick_txt = './entropy/E{}/entropyPick{}.txt'.format(self.exp, pace)
            # # if os.path.exists(mae_pick_txt) and os.path.exists(fn_predictMAE):
            # #     pass
            # # else:
            # self.predict(pace, self.soft)
            # lines = self.readtxt('./Entropy.txt')
            # self.savetxt(ent_pick_txt, lines)
           
            # mae_pick = self.readtxt(mae_pick_txt)
            # ent_pick = self.readtxt(ent_pick_txt)

            assert len(ent_pick) == len(mae_pick), 'entropy do not equal to pred %d vs %d' % (len(ent_pick), len(mae_pick))
            for i in range(len(mae_pick)):
                diff = float(mae_pick[i].split(':')[-1])
                img = mae_pick[i].split(':')[1].split(',')[0][1:]
                ent = float(ent_pick[i].split(':')[-1])
                if self.ent_threshold < 0:
                    if ent < self.ent_threshold:
                        ent = self.ent_threshold
                diff = diff - self.alpha[pace-1] * ent
                pick_soft.append((img, diff))
            pick_soft.sort(key=lambda x:x[1])
            num_pick = len(pick_soft)
            lambda_0 = pick_soft[-1][1]
            lambda_1 = pick_soft[int(num_pick*0.9)][1]
            epsilon = 1 / (1/lambda_1 - 1/lambda_0)
            pick_new = []
            for i, (img, diff) in enumerate(pick_soft):
                label = str(int(self.dict_train[img]))
                if i < num_pick*0.9:
                    weight = 10000
                else:
                    weight = int(10000*(epsilon / diff - epsilon / lambda_0))
                pick_new.append(img + ' ' + label + ' ' + str(weight) + '\n')
            pick_soft = pick_new

        if self.soft:
            fn_pick_soft = './images/txt-{}/trainPick_soft{}.txt'.format(self.exp, pace)
            self.savetxt(fn_pick_soft, pick_soft)

            print('new pick: %d' % len(pick_soft))
            print('entropy pick %d' % len(pick_ent))
            print('new left: %d' % len(left))
            if pace == 0:
                fn_left_soft = './images/txt-{}/trainLeft_soft{}.txt'.format(self.exp, pace)
                self.savetxt(fn_left_soft, left_soft)
                return (fn_left_soft, fn_pick_soft)
            else:
                return (fn_save_left, fn_pick_soft)

        else:
            print('new pick: %d' % len(pick))
            print('entropy pick %d' % len(pick_ent))
            print('new left: %d' % len(left))
            return (fn_save_left, fn_save_pick)

if __name__ == '__main__':
    percent=[0.5, 0.125, 0.125, 0.125, 0.125]
    alpha=[1.2, 0.9, 0.87, 0.85, 0.8]
    pick = PickSamples(exp=15, pace=0, percent=percent, alpha=alpha,
                        ent_threshold=-4.75, diff_threshold=6, ent_pick_per=100, root='.')
    for i in range(1,2):
        print('Pace %d: ' % i)
        a, b = pick.pick(pace=i)
        # print(a, b)

    # pick = PickSamples(exp=14.1, pace=0, percent=percent, alpha=alpha,
    #                     ent_threshold=-4.75, diff_threshold=6, ent_pick_per=0.1, root='.')

    # pick.pick(pace=1)

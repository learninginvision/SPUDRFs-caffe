
import os
import train
import predict
from picksamples import PickSamples


pace_percent = [0.5, 1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18, 1.0/18]
# pace_percent = [0.2, 0.2, 0.2, 0.2, 0.2]
# alpha=[1.2, 0.9, 0.87, 0.85, 0.8]
alpha=[15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
ent_ths = []
Exp = 1
max_step = 40000 
picksamples = PickSamples(exp=Exp, percent=pace_percent, pace=0, alpha=alpha, 
                        ent_threshold=-3.0, diff_threshold=1000, ent_pick_per=1155, 
                        random_pick=False, soft=True, root='.',max_step=max_step)

for pace in range(0, len(alpha)+1):
    print('Pace %d' % pace)
    left_txt, pick_txt = picksamples.pick(pace=pace)
    print('left_txt: %s' % left_txt)
    print('pick_txt: %s' % pick_txt)
    # break
    train_dict = {}
    train_dict['pace'] = pace
    train_dict['record'] = str(pace) + 'EPFL-VGG-NO.record'
    train_dict['data'] = str(pace) + 'EPFL-VGG-NO'
    train_dict['save'] = './checkpoints/M' + str(Exp) + '/'
    train_dict['tmp_dir'] = './tmp/Exp' + str(Exp) + '/'
    if pace == 0:
        train_dict['traintxt'] = left_txt
        train_dict['base_weights'] = '/root/data/aishijie/FGNET/model/VGG_FACE.caffemodel'
    elif pace == 1: 
        train_dict['traintxt'] = pick_txt
        train_dict['base_weights'] = '/root/data/aishijie/FGNET/model/VGG_FACE.caffemodel'
    else:
        train_dict['traintxt'] = pick_txt
        train_dict['base_weights'] = './checkpoints/M' + str(Exp) + '/' + str(pace-1) + \
            'EPFL-VGG-NO_iter_{}.caffemodel'.format(max_step)

    train_dict['testtxt'] = './images/MORPH-test.txt'

    print(train_dict)
    net = train.SPUDRFs(parser_dict=train_dict)
    net.train(pace)
    with open('./Entropy.txt', 'r') as f:
        lines = f.readlines()
    assert len(lines) > 2, 'train entropy.txt is null!'
    if not os.path.isdir('./entropy/train/'):
        os.makedirs('./entropy/train/')
    fn_newEntropy = './entropy/train/' + str(pace) + 'entropy.txt'
    with open(fn_newEntropy, 'w') as f:
        f.writelines(lines)

    with open(left_txt, 'r') as f:
        left_lines = f.readlines()
    if len(left_lines) > 0:
        pred_dict = {}
        pred_dict['test'] = './images/txt-'+ str(Exp) + '/trainLeft' + str(pace) + '-0.15.txt'
        pred_dict['predict'] = './MAE/mae' + str(Exp) + '/MAEOnTrainLeft' + str(pace) + '-0.15.txt'
        pred_dict['deploy'] = train_dict['tmp_dir'] + str(pace) + 'EPFL-VGG-NO-deploy.prototxt'
        pred_dict['model'] = train_dict['save'] + str(pace) + \
                'EPFL-VGG-NO_iter_{}.caffemodel'.format(max_step)
        diff_ave = predict.Predict(pred_dict)
        with open('./Entropy.txt', 'r') as f:
            lines = f.readlines()
        fn_save_entropy = './entropy/E' + str(Exp) + '/entropy' + str(pace) + '.txt'
        with open(fn_save_entropy, 'w') as f:
            f.writelines(lines)

        assert len(lines) > 2, 'predict entropy.txt is null!'

def init():
    train_dict = {}
    train_dict['record'] = str(pace) + 'EPFL-VGG-NO.record'
    train_dict['data'] = str(pace) + 'EPFL-VGG-NO'
    train_dict['save'] = './checkpoints/M' + str(Exp) + '/'
    train_dict['tmp_dir'] = './tmp/Exp' + str(Exp) + '/'
    train_dict['traintxt'] = './images/train_R.txt'
    train_dict['testtxt'] = './images/test_R.txt'

    train_dict['base_weights'] = '/root/data/aishijie/FGNET/model/VGG_FACE.caffemodel'

import subprocess

datasets = ['CUB', 'aircraft', 'fc100',  'omniglot',  'texture',  'traffic_sign', 'quick_draw', 'vgg_flower']
methods = ['baseline', 'baseline++', 'matchingnet', 'protonet', 'relationnet']

cmds = []
cmds.append(['nohup', 'python', '-u', './save_features.py', '--model', 'ResNet18', '--source', 'ILSVRC', '--dataset'])
cmds.append(['--target'])
cmds.append(['--method '])
cmds.append(['--train_n_way'])
cmds.append(['--test_n_way'])
cmds.append(['--n_shot'])
cmds.append(['--gpu'])
cmds.append(['> stdout 2> stderr&'])

# ILSVRC --dataset cross_CUB --target CUB --method baseline --gpu str(gpu)> stdout 2> stderr&]

gpu = 0

for way in [5, 20, 40]:
    for shot in [1, 5]:
        if way == 5 and shot ==1: continue
        for dataset in datasets:
            for method in methods:
                cmd = cmds[0] + ['cross_'+dataset] + cmds[1] + [dataset] + cmds[2] + [method]\
                      + cmds[3] + [str(way)] + cmds[4] + [str(way)] + cmds[5] + [str(shot)]\
                      + cmds[6] + [str(gpu % 8)] + cmds[7]
                print(' '.join(cmd))
                subprocess.call(cmd)
                gpu += 1
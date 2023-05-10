'''
test all pretrained models from cross validation
the pkl file should be named like kfold_x.pkl
run testset, average the probability from 5 models
store results in csv files

python test.py --model xxx
'''
import argparse
import os
import torch
import timm
from torch.utils.data import DataLoader
from dataset import dataset
import pandas as pd
import torch.nn.functional as F
import numpy as np

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet18', help='model')
parser.add_argument('--gpu', default=0, type=int, help='gpu')
parser.add_argument('--batch-size', default=32, type=int, help='batch-size')
parser.add_argument('--load-dir', default='checkpoints', type=str, help='where to load model')
parser.add_argument('--save-dir', default='results', type=str, help='where to save csv file')
args = parser.parse_args()

# dataset
testset = dataset(test=True)
probListSum = np.zeros((len(testset),3))
testloader = DataLoader(testset, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True)
nameList = []
for img, label, name in testloader:
    nameList.extend(name)

for i in range(5):
    # backbone network
    if args.model == 'resnet18':
        net = timm.create_model('resnet18', pretrained=True, num_classes=3).to(args.gpu)
    elif args.model == 'resnet50d':
        net = timm.create_model('resnet50d', pretrained=True, num_classes=3).to(args.gpu)
    elif args.model == 'incepv3':
        net = timm.create_model('inception_v3', pretrained=True, num_classes=3).to(args.gpu)
    elif args.model == 'effb2':
        net = timm.create_model('tf_efficientnet_b2', pretrained=True, num_classes=3).to(args.gpu)

    # load pretrained model
    loadpath = os.path.join(args.load_dir,args.model,f'kfold_{i}.pkl')
    ckpt = torch.load(loadpath)
    state_dict = ckpt['state']
    kappa = ckpt['kappa']
    epoch = ckpt['epoch']
    # if epoch == -1:
    #     print(f'Loading model from {loadpath} which reaches kappa {kappa} in cross validation')
    # else:
    print(f'Loading model from {loadpath} which reaches kappa {kappa} at epoch {epoch}...')
    net.load_state_dict(state_dict)
    net.eval()

    savepath = os.path.join(args.save_dir,args.model,f'kfold_{i}.csv')

    probList = []
    classList = []
    with torch.no_grad():
        print('testing...')
        for img, label, name in testloader:
            img = img.to(args.gpu)
            label_pred = net(img).cpu()
            classList.extend(label_pred.max(1)[1].tolist()) 
            probs = F.softmax(label_pred,dim=1)
            probList.extend(probs.tolist())

        print(f'Saving results to {savepath}')
        npproblist = np.array(probList)
        dataframe = pd.DataFrame({'case':nameList,'class':classList,'P0':npproblist[:,0],'P1':npproblist[:,1],'P2':npproblist[:,2]})
        dataframe.to_csv(savepath, header=True, index=None)
        
        probListSum += npproblist


savepath = os.path.join(args.save_dir,args.model,f'avg.csv')
print(f'Saving avg results to {savepath}')
probListAvg = probListSum/5
labels = np.argmax(probListAvg, axis=1)
dataframe = pd.DataFrame({'case':nameList,'class':labels,'P0':probListAvg[:,0],'P1':probListAvg[:,1],'P2':probListAvg[:,2]})
dataframe.to_csv(savepath, header=True, index=None)

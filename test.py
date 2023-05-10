'''
test one pretrained model
run testset
store result in csv file

python test.py --model xxx -file-name xxx.pkl
'''
import argparse
import os
import torch
import timm
from torch.utils.data import DataLoader
from dataset import dataset
import pandas as pd
import torch.nn.functional as F

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet18', help='model')
parser.add_argument('--gpu', default=0, type=int, help='gpu')
parser.add_argument('--batch-size', default=32, type=int, help='batch-size')
parser.add_argument('--load-dir', default='checkpoints', type=str, help='where to load model')
parser.add_argument('--file-name', default='kfold_0.pkl', type=str, help='file name')
parser.add_argument('--save-dir', default='results', type=str, help='where to save csv file')
args = parser.parse_args()

# backbone network
if args.model == 'resnet18':
    net = timm.create_model('resnet18', pretrained=True, num_classes=3).to(args.gpu)
elif args.model == 'resnet50d':
    net = timm.create_model('resnet50d', pretrained=True, num_classes=3).to(args.gpu)

# load pretrained model
loadpath = os.path.join(args.load_dir,args.model,args.file_name)
ckpt = torch.load(loadpath)
state_dict = ckpt['state']
kappa = ckpt['kappa']
epoch = ckpt['epoch']
if epoch == -1:
    print(f'Loading model from {loadpath} which reaches kappa {kappa} in cross validation')
else:
    print(f'Loading model from {loadpath} which reaches kappa {kappa} at epoch {epoch}...')
net.load_state_dict(state_dict)
net.eval()

# dataset
testset = dataset(test=True)
testloader = DataLoader(testset, shuffle=False, batch_size=args.batch_size, num_workers=4, pin_memory=True)

savepath = os.path.join(args.save_dir,args.model,args.file_name[:-3]+'csv')

newflag = True
with torch.no_grad():
    print('testing...')
    for img, label, name in testloader:
        img = img.to(args.gpu)
        label_pred = net(img).cpu()
        classList = label_pred.max(1)[1]
        probs = F.softmax(label_pred,dim=1)
        dataframe = pd.DataFrame({'case':name,'class':classList,'P0':probs[:,0],'P1':probs[:,1],'P2':probs[:,2]})
        if newflag:
            dataframe.to_csv(savepath, header=True, index=None)
            newflag = False
        else:
            dataframe.to_csv(savepath, mode='a', header=False, index=None)
    print(f'Saving results to {savepath}')

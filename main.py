import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report
import pdb
import copy

from dataset.CramedDataset import load_cremad
from dataset.VGGSoundDataset import VGGSound
from dataset.dataset import AVDataset
from models.basic_model import AVClassifier
from utils.utils import setup_seed, weight_init
from tqdm import tqdm


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='CREMAD', type=str,
                        help='VGGSound, KineticSound, CREMAD, AVE')
    parser.add_argument('--modulation', default='OGM_GE', type=str,

                        choices=['Normal', 'OGM', 'OGM_GE'])
    parser.add_argument('--fusion_method', default='concat', type=str,
                        choices=['sum', 'concat', 'gated', 'film'])
    parser.add_argument('--fps', default=1, type=int)
    parser.add_argument('--audio_path', default='/home/hudi/data/CREMA-D/AudioWAV', type=str)
    parser.add_argument('--visual_path', default='/home/hudi/data/CREMA-D/', type=str)

    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--optimizer', default='sgd', type=str, choices=['sgd', 'adam'])
    parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr_decay_step', default=70, type=int, help='where learning rate decays')
    parser.add_argument('--lr_decay_ratio', default=0.1, type=float, help='decay coefficient')

    parser.add_argument('--ckpt_path', default="/ckpt", type=str, help='path to save trained models')
    parser.add_argument('--train', action='store_true', help='turn on train mode')

    parser.add_argument('--use_tensorboard', default=False, type=bool, help='whether to visualize')
    parser.add_argument('--tensorboard_path', type=str, help='path to save tensorboard logs')

    parser.add_argument('--random_seed', default=0, type=int)
    return parser.parse_args()


def train_epoch(args, epoch, model, device, dataloader, optimizer, scheduler, writer=None):
    criterion = nn.CrossEntropyLoss()

    model.train()
    print("Start training ... ")

    _loss = 0

    for step, (spec, image, label) in tqdm(enumerate(dataloader), desc='Epoch: {}: '.format(epoch)):
        #pdb.set_trace()
        spec = spec.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        # TODO: make it simpler and easier to extend
        a, v, out = model(spec.unsqueeze(1).float(), image.float())

        loss = criterion(out, label)
        loss.backward()

        optimizer.step()

        _loss += loss.item()

    scheduler.step()

    return _loss / len(dataloader)


def eval(args, model, device, dataloader, test=False):
    softmax = nn.Softmax(dim=1)

    if args.dataset == 'CREMAD':
        n_classes = 6
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    with torch.no_grad():
        model.eval()
        criterion = nn.CrossEntropyLoss()
        _loss = 0
        golds = []
        preds = []
        for step, (spec, image, label) in enumerate(dataloader):

            spec = spec.to(device)
            image = image.to(device)
            label = label.to(device)

            a, v, out = model(spec.unsqueeze(1).float(), image.float())


            loss = criterion(out, label)
            _loss += loss.item()

            y_hat = torch.argmax(softmax(out), dim=-1)
            golds.extend(label.cpu().numpy())
            preds.extend(y_hat.cpu().numpy())

        wf1 = f1_score(golds, preds, average='weighted')
        
        if test:
            print(classification_report(golds, preds))
    return _loss / len(dataloader), wf1


def main():
    args = get_arguments()
    print(args)

    setup_seed(args.random_seed)
 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = AVClassifier(args)
    model.apply(weight_init)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_decay_step, args.lr_decay_ratio)

    if args.dataset == 'CREMAD':
        train_dataset, dev_dataset, test_dataset = load_cremad(args)
    else:
        raise NotImplementedError('Incorrect dataset name {}'.format(args.dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=True)

    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size,
                                shuffle=False, pin_memory=True)

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=True)

    if args.train:

        best_dev_f1 = 0.0
        best_state = None
        best_epoch = 0

        for epoch in range(args.epochs):
            batch_loss= train_epoch(args, epoch, model, device,
                                        train_dataloader, optimizer, scheduler)
            loss, f1 = eval(args, model, device, dev_dataloader)
        
            print('Epoch: {}, Train Loss: {}, Dev Loss: {}, F1: {}'.format(epoch, batch_loss, loss, f1))

            if f1 > best_dev_f1:
                best_dev_f1 = f1
                best_state = best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch
        
        # Best state
        model.load_state_dict(best_state)
        print('Best model loaded at epoch {} with dev F1: {}'.format(best_epoch, best_dev_f1))
        print("Start testing on test dataset ...")
        _, f1 = eval(args, model, device, test_dataloader, test=True)
        if not os.path.exists(args.ckpt_path):
            os.mkdir(args.ckpt_path)

        model_name = 'best_model_of_dataset_{}_' \
                        'optimizer_{}_' \
                        'epoch_{}_f1_{}.pth'.format(args.dataset,
                                                    args.optimizer,
                                                    epoch, f1)

        saved_dict = {'saved_epoch': epoch,
                        'fusion': args.fusion_method,
                        'f1': f1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()}

        save_dir = os.path.join(args.ckpt_path, model_name)

        torch.save(saved_dict, save_dir)
        print('The best model has been saved at {}.'.format(save_dir))

                

    else:
        loaded_dict = torch.load(args.ckpt_path)
        fusion = loaded_dict['fusion']
        state_dict = loaded_dict['model']

        assert fusion == args.fusion_method, 'inconsistency between fusion method of loaded model and args !'

        model.load_state_dict(state_dict)
        print('Trained model loaded! Testing ...')

        loss, f1 = eval(args, model, device, test_dataloader, test=True)
        


if __name__ == "__main__":
    main()

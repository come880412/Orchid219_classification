import argparse
from torch.utils.data import DataLoader
import torch
import tqdm
import os
import numpy as np
import csv
import torch.nn as nn

from dataset import Orchid_data
from utils import get_acc, get_f1
from model import *
from transformers import ViTForImageClassification, BeitForImageClassification

from timm.models.convnext import convnext_xlarge_384_in22ft1k
from timm.models.swin_transformer import swin_large_patch4_window12_384_in22k

import warnings
warnings.filterwarnings("ignore")

def test(opt, model, criterion, val_loader):
    model.eval()

    y_true = torch.tensor([]).type(torch.int16)
    y_pred = torch.tensor([]).type(torch.int16)
    total_correct = 0
    total_label = 0
    val_loss = 0.
    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="%s" % opt.mode, unit=" step")

    save_list = [['filename','category']]
    for images, image_640, labels, data_names in val_loader:
        with torch.no_grad():
            images, image_640, labels = images.cuda(), image_640.cuda(), labels.cuda()

            if opt.model == 'ViT' or opt.model == 'beit':
                preds = model(images)
                preds = preds.logits
            elif opt.model == 'cnn_swin_fusion':
                out_CNN, out_SWIN, emb_fusion_out = model(images)
                preds = (out_CNN + out_SWIN + emb_fusion_out) / 3
                loss = criterion(preds, labels)
            elif opt.model == 'ensemble':
                preds = model(images, image_640)
            else:
                preds, _ = model(images)

            loss = criterion(preds, labels)
            correct, total = get_acc(preds, labels)

            total_label += total
            total_correct += correct
            val_acc = (total_correct / total_label) * 100

            val_loss += loss

            labels = labels.cpu().detach()
            preds = preds.cpu().detach()
            y_true = torch.cat((y_true, labels), 0)
            y_pred = torch.cat((y_pred, preds), 0)

            pbar.update()
            pbar.set_postfix(
                loss=f"{val_loss:.4f}",
                Accuracy=f"{val_acc:.2f}"
            )

            preds = preds.numpy()
            labels = labels.numpy()

            pred_label = np.argmax(preds, axis=1)
            for i in range(len(data_names)):
                data_name = data_names[i]
                pred = pred_label[i]

                save_list.append([data_name, str(pred)])
            
    Macro_f1 = get_f1(y_pred, y_true)
    Final_score = val_acc / 100 * 0.5 + Macro_f1 * 0.5

    pbar.set_postfix(
        loss=f"{val_loss:.4f}",
        Accuracy=f"{val_acc:.2f}%",
        F1=f"{Macro_f1:.3f}",
        Final=f"{Final_score:.4f}"
    )
    pbar.close()
    # print(f1_dict)
    
    np.savetxt(os.path.join(opt.save_path, '%s.csv' % opt.mode),  save_list, fmt='%s', delimiter=',')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../dataset', help='path to dataset')
    parser.add_argument('--num_classes', type=int, default=219, help='number of classes')

    parser.add_argument('--mode', type=str, default="test", help='valid/test')
    parser.add_argument('--split_type', default='random', help='random, k_fold')
    parser.add_argument('--train_type', default='pretrain', help='pretrain/fine_tune')
    parser.add_argument('--load', default='./public_model/beit_best.pth', help='path to model to continue training')
    parser.add_argument('--load_pretrain', default='', help='path to model to continue training')
    parser.add_argument('--k', default='5', help='Which fold you want to use')
    parser.add_argument('--save_path', type=str, default="./output_csv")

    parser.add_argument("--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument('--model', default='beit', help='resnet50/convnext/pvt/ViT/beit/swin/ensemble/cnn_swin_fusion/crop_fusion_att')
    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu workers')
    
    opt = parser.parse_args()
    os.makedirs(opt.save_path, exist_ok=True)
    print(f"model: {opt.model}    split_type:{opt.split_type}")

    test_data = Orchid_data(opt, opt.mode)
    test_loader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu, drop_last=False)

    if opt.model == 'ensemble':
        model_list = []
        model_weight_path = './public_model/beit_best.pth'
        model = BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-384')
        model.classifier = nn.Linear(model.classifier.in_features, out_features=opt.num_classes)
        model.load_state_dict(torch.load(model_weight_path))
        model.eval()
        model = model.cuda()
        model_list.append(model)

        model_weight_path = './public_model/swin_best.pth'
        model = swin_large_patch4_window12_384_in22k(False)
        model.head = nn.Linear(in_features=model.head.in_features, out_features=opt.num_classes)
        model.load_state_dict(torch.load(model_weight_path))
        model.eval()
        model = model.cuda()
        model_list.append(model)
        
        model_weight_path = './public_model/convnext_xlarge.pth'
        # model_weight_path = './checkpoints/convnext_xlarge_size640_orchid219/random/model_best.pth'
        model = convnext_xlarge_384_in22ft1k(False)
        model.head.fc = nn.Linear(in_features=model.head.fc.in_features, out_features=opt.num_classes)
        model.load_state_dict(torch.load(model_weight_path))
        model.eval()
        model = model.cuda()
        model_list.append(model)

        model = ensemble_net(model_list)
    else:
        if opt.model == 'resnet50':
            model = resnet50(opt)
        elif opt.model == 'convnext':
            model = convnext_xlarge_384_in22ft1k(False)
            model.head.fc = nn.Linear(in_features=model.head.fc.in_features, out_features=opt.num_classes)
        elif opt.model == 'swin':
            model = swin_large_patch4_window12_384_in22k(False)
            model.head = nn.Linear(in_features=model.head.in_features, out_features=opt.num_classes)
        elif opt.model == 'ViT':
            model = ViTForImageClassification.from_pretrained('google/vit-large-patch32-384')
            model.classifier = nn.Linear(model.classifier.in_features, out_features=opt.num_classes)
        elif opt.model == 'beit':
            model = BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-384')
            model.classifier = nn.Linear(model.classifier.in_features, out_features=opt.num_classes)
        elif opt.model == 'pvt':
            model = pvt_v2_b5(True)
            model.load_state_dict(torch.load('./pretrain/pvt_v2_b5.pth', map_location='cpu'))
            model.head = nn.Linear(model.head.in_features, opt.num_classes)
        elif opt.model == 'crop_fusion_att':
            model = crop_fusion_att(opt)
        elif opt.model == 'cnn_swin_fusion':
            model = cnn_swin_fusion(opt)

        if opt.load != '':
            print(f'loading pretrained model from {opt.load}')
            model.load_state_dict(torch.load(opt.load))
        model = model.cuda()
        model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    test(opt, model, criterion, test_loader)
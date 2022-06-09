import argparse
from torch.utils.data import DataLoader
import torch
import tqdm
import os
import numpy as np
import csv
import torch.nn as nn

from dataset import Orchid_public
from model import *
from transformers import ViTForImageClassification, BeitForImageClassification

from timm.models.convnext import convnext_xlarge_384_in22ft1k
from timm.models.swin_transformer import swin_large_patch4_window12_384_in22k

import warnings
warnings.filterwarnings("ignore")

def public(opt, model, public_loader):
    model.eval()

    pbar = tqdm.tqdm(total=len(public_loader), ncols=0, desc="public", unit=" step")

    save_list = [['filename','category']]
    for images, image_640, image_names in public_loader:
        with torch.no_grad():
            images, image_640 = images.cuda(), image_640.cuda()

            if opt.model == 'ViT' or opt.model == 'beit':
                preds = model(images)
                preds = preds.logits
            elif opt.model == 'cnn_swin_fusion':
                out_CNN, out_SWIN, emb_fusion_out = model(images)
                preds = (out_CNN + out_SWIN + emb_fusion_out) / 3
            elif opt.model == 'ensemble':
                preds = model(images)
            else:
                preds, _ = model(images)

            preds = preds.cpu().detach().numpy()

            pbar.update()

            pred_label = np.argmax(preds, axis=1)
            for i in range(len(image_names)):
                image_name = image_names[i]
                pred = pred_label[i]

                save_list.append([image_name, str(pred)])
            
    pbar.close()
    
    print("-------Save csv----------")
    np.savetxt(os.path.join(opt.save_path, '%s.csv' % opt.model) ,  save_list, fmt='%s', delimiter=',')
    print("-------Finished!----------")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='../dataset', help='path to dataset')
    parser.add_argument('--num_classes', type=int, default=219, help='number of classes')

    parser.add_argument('--load', default='./public_model/swin_best.pth', help='path to model to continue training')
    parser.add_argument('--load_pretrain', default='', help='path to model to continue training')
    parser.add_argument('--save_path', type=str, default="./output_csv")

    parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
    parser.add_argument('--model', default='swin', help='resnet50/convnext/pvt/ViT/beit/swin/ensemble/cnn_swin_fusion/crop_fusion_att')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu workers')
    
    opt = parser.parse_args()
    os.makedirs(opt.save_path, exist_ok=True)

    public_data = Orchid_public(opt)
    public_loader = DataLoader(public_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu, drop_last=False)

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

    public(opt, model, public_loader)
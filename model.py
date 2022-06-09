from turtle import forward
import torchvision.models as models
import torch.nn as nn
import torch
from torchsummary import summary
import torch.nn.functional as F

import argparse

from model_component.Resnet import ResNet, Bottleneck
from model_component.PVT import *
from timm.models.convnext import convnext_xlarge_384_in22ft1k
from timm.models.swin_transformer import swin_large_patch4_window12_384_in22k

class ensemble_net(nn.Module):
    def __init__(self, model_list):
        super().__init__()

        self.model_list = model_list
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        out = []
        for i, model in enumerate(self.model_list): 
            try: # convnext and swin
                pred, _ = model(x)
                out.append(self.softmax(pred))
            except: # beit
                pred = model(x)
                pred = pred.logits
                out.append(self.softmax(pred))
        out = torch.stack(out, dim=2)

        return torch.mean(out, dim=2)

class resnet50(nn.Module):
    def __init__(self, opt):
        super(resnet50, self).__init__()
        self.train_type = opt.train_type
        self.load_pretrain = opt.load_pretrain

        model = ResNet(block = Bottleneck, layers = [3, 4, 6, 3], num_classes = opt.num_classes)
        if opt.load_pretrain:
            dict_ = torch.load(self.load_pretrain)
            model_dict = model.state_dict()
            pretrained_dict = {}
            for k, v in dict_.items():
                if k.split('.')[1] != 'classifier':
                    pretrained_dict.update({k[6:]: v})
                else:
                    break
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
        else:
            dict_ = models.resnet50(pretrained = True)
            dict_ = dict_.state_dict()
            model_dict = model.state_dict()
            pretrained_dict = {}
            for k, v in dict_.items():
                if k in model_dict and k.split('.')[0] != 'fc':
                    pretrained_dict.update({k: v})
                else:
                    break
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            
        self.model = model
    def forward(self, x):
        out, emb = self.model(x)
        return out, emb

class attention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.weight = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(in_features= dim , out_features= 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
            input :
                x : (B, n, dim)
            output :
                out : (B, dim)
        """
        _, _, seq_len = x.size()

        att = []
        for i in range(0, seq_len, 1):
            att_weight = self.weight(x[:, :, i]) # (B, 1)
            att.append(att_weight)

        att = torch.stack(att, dim=2)

        return att
            
class crop_fusion_att(nn.Module):
    def __init__(self, opt):
        super(crop_fusion_att,self).__init__()
        self.num_classes = opt.num_classes
        
        self.Feature_Extractor = convnext_xlarge_384_in22ft1k(False)
        self.Feature_Extractor.head.fc = nn.Linear(in_features=self.Feature_Extractor.head.fc.in_features, out_features=opt.num_classes)
        if opt.load_pretrain:
            self.Feature_Extractor.load_state_dict(torch.load(opt.load_pretrain))
        self.d_model = 2048 # convnext:2048

        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.global_attention = attention(self.d_model)
        self.relation_attention = attention(2 * self.d_model)

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=2 * self.d_model, out_features=self.d_model),
            nn.BatchNorm1d(self.d_model),
            nn.ReLU(),
            nn.Linear(in_features=self.d_model, out_features=self.num_classes)
        )
    
    def forward(self, x):
        cropped_image = self.crop_image(x)

        emb_out = []
        for frame_idx in range(cropped_image.shape[1]):
            frame = cropped_image[:, frame_idx, :, :, :]
            _, embedding = self.Feature_Extractor(frame)
            emb = self.GAP(embedding[-1]).squeeze()

            if emb.dim() == 1: # If batch_size = 1
                emb = emb.unsqueeze(0)

            emb_out.append(emb)
        
        _, embedding = self.Feature_Extractor(x)
        emb = self.GAP(embedding[-1]).squeeze()
        if emb.dim() == 1: # If batch_size = 1
            emb = emb.unsqueeze(0)

        emb_out.append(emb)
        emb_out = torch.stack(emb_out, dim = 2)
        # emb_out : (B, d_model, num_images)
        emb_ori_image = emb_out[:, :, -1]
        emb_crop_image = emb_out[:, :, :-1]

        att_1 = self.global_attention(emb_crop_image)
        crop_global_emb = emb_crop_image.mul(att_1).sum(2).div(att_1.sum(2))

        emb_cat = []
        for i in range(emb_out.shape[2]):
            emb_cat.append(torch.cat((emb_out[:, :, i], crop_global_emb), dim=1))
        emb_cat = torch.stack(emb_cat, dim=2)

        att_2 = self.relation_attention(emb_cat)

        out = emb_cat.mul(att_2).sum(2).div(att_2.sum(2))
        out = self.classifier(out)

        return out, att_2

    def crop_image(self, image):
        b, _, h, w = image.size()
        crop_h, crop_w = h // 6, w // 6
        image_list = []
        
        for i in range(h // crop_h):
            for j in range(w // crop_w):
                cropped_image = image[:, :, i*crop_h:(i+1)*crop_h, j*crop_w:(j+1) * crop_w]
                image_list.append(cropped_image)

        return torch.stack(image_list, dim=1)

class cnn_swin_fusion(nn.Module):
    def __init__(self, opt):
        super(cnn_swin_fusion, self).__init__()
        self.CNN = convnext_xlarge_384_in22ft1k(False)
        self.SWIN = swin_large_patch4_window12_384_in22k(False)
        self.CNN.head.fc = nn.Linear(in_features=self.CNN.head.fc.in_features, out_features=opt.num_classes)
        self.SWIN.head = nn.Linear(in_features=self.SWIN.head.in_features, out_features=opt.num_classes)
        
        if opt.load_pretrain:
            self.CNN.load_state_dict(torch.load('./pretrain/convnext_xlarge.pth'))
            self.SWIN.load_state_dict(torch.load('./pretrain/BestSwin_large_patch4_window12_384_in22k_mixup_0.1.pth'))

        self.GAP = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(8064, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Linear(512, opt.num_classes),
        )
        
    def forward(self, x):
        out_CNN, emb_CNN = self.CNN(x)
        out_SWIN, emb_SWIN = self.SWIN(x)
        for i in range(len(emb_CNN)):
            emb_cnn = self.GAP(emb_CNN[i]).squeeze()
            emb_swin = emb_SWIN[i]
            
            emb_concat = torch.concat((emb_cnn, emb_swin), dim=1)
            if i == 0:
                emb = emb_concat
            else:
                emb = torch.concat((emb, emb_concat), dim=1)
        output = self.classifier(emb)
    
        return out_CNN, out_SWIN, output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument('--warmup_epochs', default=10, type=int, help='number of warmup epochs')
    parser.add_argument("--initial_epoch", type=int, default=0, help="Start epoch")

    parser.add_argument('--root', default='../dataset/Orchid219', help='path to dataset')
    parser.add_argument('--train_type', default='fine_tune', help='pretrain/fine_tune')
    parser.add_argument('--load_pretrain', default='./checkpoints/convnext_pretrain/model_best.pth', help='path to model to continue training')

    parser.add_argument('--split_type', default='random', help='random, k_fold')
    parser.add_argument('--k', default='1', help='Which fold you want to use')
    
    parser.add_argument('--num_classes', type=int, default=219, help='number of classes')
    parser.add_argument('--optimizer', default='adamw', help='adamw/sgd')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--mixup", type=float, default=0.4, help="mixup alpha")
    parser.add_argument("--smooth_factor", type=float, default=0.1, help="The factor of label_smoothing")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay factor")
    parser.add_argument("--batch_size", type=int, default=2, help="batch_size")

    parser.add_argument('--model', default='pvt', help='resnet50/convnext/ViT/beit/RepLKNet/pvt')
    parser.add_argument('--scheduler', type=str, default='cosine', help='cosine/linearwarmup')

    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu workers')
    parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')

    parser.add_argument('--load', default='', help='path to model to continue training')
    parser.add_argument('--save_model', default='./checkpoints', help='path to save model')
    parser.add_argument('--save_name', default='beit_orchid219', help='Name of saving model')
    
    opt = parser.parse_args()

    x = torch.randn((2, 3, 128, 128)).cuda()
    # model = resnet50(opt, pretrained=True).cuda()
    # model = pvt_v2_b5(True).cuda()
    model = cnn_pvt_fusion(opt).cuda()
    # model = crop_fusion_att(opt).cuda()
    # summary(model, (3,384,384))
    # summary(pvt_v2_b4(True).cuda(), (3,384,384))
    
    out, emb = model(x)
    print(out.shape)


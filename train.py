import argparse
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import tqdm
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy

from dataset import Orchid_data, Orchid_pretrain_data
from utils import *
from model import *
from transformers import ViTForImageClassification, BeitForImageClassification
from model_component.RepLKNet import *
from model_component.PVT import *

import warnings
warnings.filterwarnings("ignore")

def train_batch(opt, model, optimizer, criterion, image, label):
    optimizer.zero_grad()

    if opt.model == 'ViT' or opt.model == 'beit':
        pred = model(image)
        pred = pred.logits
        loss = criterion(pred, label)

    elif opt.model == 'cnn_swin_fusion':
        out_CNN, out_SWIN, emb_fusion_out = model(image)
        pred = (out_CNN + out_SWIN + emb_fusion_out) / 3
        loss = criterion(pred, label)
    else:
        pred, emb = model(image)
        loss = criterion(pred, label)

    loss.backward()
    optimizer.step()

    return loss, pred

def validation(opt, model, val_loader, writer, epoch):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    y_true = torch.tensor([]).type(torch.int16)
    y_pred = torch.tensor([]).type(torch.int16)
    total_correct = 0
    total_label = 0
    val_loss = 0.
    pbar = tqdm.tqdm(total=len(val_loader), ncols=0, desc="val", unit=" step")
    for image, label, _ in val_loader:
        with torch.no_grad():
            image, label = image.cuda(), label.cuda()

            if opt.model == 'ViT' or opt.model == 'beit':
                pred = model(image)
                pred = pred.logits
                loss = criterion(pred, label)
                
            elif opt.model == 'cnn_swin_fusion':
                out_CNN, out_SWIN, emb_fusion_out = model(image)
                pred = (out_CNN + out_SWIN + emb_fusion_out) / 3
                loss = criterion(pred, label)
            else:
                pred, emb = model(image)
                loss = criterion(pred, label)

            correct, total = get_acc(pred, label)

            total_label += total
            total_correct += correct
            val_acc = (total_correct / total_label) * 100

            val_loss += loss

            label = label.cpu().detach()
            pred = pred.cpu().detach()
            y_true = torch.cat((y_true, label), 0)
            y_pred = torch.cat((y_pred, pred), 0)

            pbar.update()
            pbar.set_postfix(
                loss=f"{val_loss:.4f}",
                Accuracy=f"{val_acc:.2f}%"
            )
    
    Macro_f1 = get_f1(y_pred, y_true)
    Final_score = val_acc / 100 * 0.5 + Macro_f1 * 0.5

    pbar.set_postfix(
        loss=f"{val_loss:.4f}",
        Accuracy=f"{val_acc:.2f}%",
        F1=f"{Macro_f1:.3f}",
        Final=f"{Final_score:.4f}"
    )
    pbar.close()
    
    writer.add_scalar('validation loss', val_loss, epoch)
    writer.add_scalar('validation accuracy', val_acc, epoch)
    writer.add_scalar('validation F1', Macro_f1, epoch)
    writer.add_scalar('validation final score', Final_score, epoch)

    return Final_score

def main(opt, model, criterion, optimizer, scheduler, train_loader, val_loader, mixup_fn):
    if opt.split_type == 'random':
        writer = SummaryWriter('runs/%s/%s' % (opt.save_name, opt.split_type))
    else:
        writer = SummaryWriter('runs/%s/%s_%s' % (opt.save_name, opt.split_type, opt.k))
    
    criterion = criterion.cuda()
    model = model.cuda()

    """training"""
    print('Start training!')
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)
    max_final_score = 0.
    train_update = 0

    for epoch in range(opt.initial_epoch, opt.n_epochs):
        model.train(True)
        pbar = tqdm.tqdm(total=len(train_loader), ncols=0, desc="Train[%d/%d]"%(epoch, opt.n_epochs), unit=" step")

        total_loss = 0
        total_correct = 0
        total_label = 0
        y_true = torch.tensor([]).type(torch.int16)
        y_pred = torch.tensor([]).type(torch.int16)

        for image, label, _ in train_loader:
            image, label = image.cuda(), label.cuda()

            target = label
            if mixup_fn is not None:
                image, label = mixup_fn(image, label)
                target = torch.argmax(label, dim=1)
            train_loss, pred = train_batch(opt, model, optimizer, criterion, image, label)

            correct, total = get_acc(pred, target)

            total_label += total
            total_correct += correct
            acc = (total_correct / total_label) * 100

            total_loss += train_loss

            target = target.cpu().detach()
            pred = pred.cpu().detach()
            y_true = torch.cat((y_true, target), 0)
            y_pred = torch.cat((y_pred, pred), 0)
        
            pbar.update()
            pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                Accuracy=f"{acc:.2f}%"
            )

            writer.add_scalar('training loss', train_loss, train_update)
            writer.add_scalar('training accuracy', acc, train_update)
            train_update += 1

        Macro_f1 = get_f1(y_pred, y_true)
        Final_score = acc / 100 * 0.5 + Macro_f1 * 0.5
        pbar.set_postfix(
                loss=f"{total_loss:.4f}",
                Accuracy=f"{acc:.2f}%",
                F1=f"{Macro_f1:.3f}",
                Final=f"{Final_score:.4f}"
            )

        pbar.close()

        val_final_score = validation(opt, model, val_loader, writer, epoch)
        if max_final_score <= val_final_score:
            print('save model!!')
            max_final_score = val_final_score
            if opt.train_type == 'fine_tune':
                if opt.split_type == 'random':
                    torch.save(model.state_dict(), os.path.join(opt.save_model, opt.save_name, opt.split_type, 'model_best.pth'))
                else:
                    torch.save(model.state_dict(), os.path.join(opt.save_model, opt.save_name, f'fold{opt.k}' , 'model_best.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(opt.save_model, opt.save_name, 'model_best.pth'))
        if opt.train_type == 'fine_tune':
            if opt.split_type == 'random':
                torch.save(model.state_dict(), os.path.join(opt.save_model, opt.save_name, opt.split_type, 'model_final.pth'))
            else:
                torch.save(model.state_dict(), os.path.join(opt.save_model, opt.save_name, f'fold{opt.k}' , 'model_final.pth'))
        else:
            torch.save(model.state_dict(), os.path.join(opt.save_model, opt.save_name, 'model_final.pth'))

        scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    print('best final score:%.2f' % (max_final_score))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument('--warmup_epochs', default=10, type=int, help='number of warmup epochs')
    parser.add_argument("--initial_epoch", type=int, default=0, help="Start epoch")

    parser.add_argument('--root', default='../dataset/Orchid219', help='path to dataset')
    parser.add_argument('--train_type', default='fine_tune', help='pretrain/fine_tune')
    parser.add_argument('--load_pretrain', default='./pretrain/convnext_xlarge.pth', help='path to model to continue training')

    parser.add_argument('--split_type', default='random', help='random, k_fold')
    parser.add_argument('--k', default='1', help='Which fold you want to use')
    
    parser.add_argument('--num_classes', type=int, default=219, help='number of classes')
    parser.add_argument('--optimizer', default='adamw', help='adamw/sgd')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--mixup", type=float, default=0.4, help="mixup alpha")
    parser.add_argument("--smooth_factor", type=float, default=0.1, help="The factor of label_smoothing")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="weight decay factor")
    parser.add_argument("--batch_size", type=int, default=2, help="batch_size")
    parser.add_argument("--seed", type=int, default=1004, help="batch_size")

    parser.add_argument('--model', default='cnn_swin_fusion', help='resnet50/convnext/ViT/beit/pvt/crop_fusion_att/cnn_swin_fusion')
    parser.add_argument('--scheduler', type=str, default='linearwarmup', help='cosine/linearwarmup')

    parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu workers')
    parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')

    parser.add_argument('--load', default='', help='path to model to continue training')
    parser.add_argument('--save_model', default='./checkpoints', help='path to save model')
    parser.add_argument('--save_name', default='beit_orchid219', help='Name of saving model')
    
    opt = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    fixed_seed(opt.seed)
    
    if opt.train_type == 'pretrain':
        print("Model pretrained on orchid_extra dataset")
        print("Model: ", opt.model)
        os.makedirs(os.path.join(opt.save_model, opt.save_name), exist_ok=True)

        train_data = Orchid_pretrain_data(opt, 'train')
        train_loader = DataLoader(train_data, batch_size=opt.batch_size,shuffle=True, num_workers=opt.n_cpu, drop_last=True)
        val_data = Orchid_pretrain_data(opt, 'valid')
        val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)
        opt.num_classes = 156

    elif opt.train_type == 'fine_tune':
        if opt.split_type == 'random':
            os.makedirs(os.path.join(opt.save_model, opt.save_name, opt.split_type), exist_ok=True)
        else:
            os.makedirs(os.path.join(opt.save_model, opt.save_name, f'fold{opt.k}'), exist_ok=True)
        if opt.split_type == 'random':
            print("Model: ", opt.model, '  Split_type:', opt.split_type)
        else:
            print("Model: ", opt.model, '  Split_type:Fold_%s' % opt.k)
        train_data = Orchid_data(opt, 'train')
        train_loader = DataLoader(train_data, batch_size=opt.batch_size,shuffle=True, num_workers=opt.n_cpu, drop_last=True)
        val_data = Orchid_data(opt, 'valid')
        val_loader = DataLoader(val_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu, drop_last=False)
        opt.num_classes = 219
        if opt.load_pretrain:
            print("Load pretrained models on Orchid156!")

    if opt.model == 'resnet50':
        model = resnet50(opt)
    elif opt.model == 'convnext':
        model = convnext_xlarge_384_in22ft1k(False)
        model.head.fc = nn.Linear(in_features=model.head.fc.in_features, out_features=opt.num_classes)
    elif opt.model == 'ViT':
        model = ViTForImageClassification.from_pretrained('google/vit-large-patch32-384')
        if opt.load_pretrain:
            model.classifier = nn.Linear(model.classifier.in_features, out_features=156)
            model.load_state_dict(torch.load(opt.load_pretrain))
        model.classifier = nn.Linear(model.classifier.in_features, out_features=opt.num_classes)
    elif opt.model == 'beit':
        model = BeitForImageClassification.from_pretrained('microsoft/beit-large-patch16-384')
        if opt.load_pretrain:
            model.classifier = nn.Linear(model.classifier.in_features, out_features=156)
            model.load_state_dict(torch.load(opt.load_pretrain))
        model.classifier = nn.Linear(model.classifier.in_features, out_features=opt.num_classes)
        
    elif opt.model == 'pvt':
        model = pvt_v2_b5(True)
        model.load_state_dict(torch.load('./pretrain/pvt_v2_b5.pth', map_location='cpu'))
        if opt.load_pretrain:
            model.head = nn.Linear(model.head.in_features, out_features=156)
            model.load_state_dict(torch.load(opt.load_pretrain))
        model.head = nn.Linear(model.head.in_features, out_features=opt.num_classes)
    elif opt.model == 'RepLKNet':
        model = create_RepLKNet31B(small_kernel_merged=False)
        model.load_state_dict(torch.load('./pretrain/RepLKNet-31B_ImageNet-22K-to-1K_384.pth', map_location='cpu'))
        model.head = nn.Linear(model.head.in_features, opt.num_classes)
        model.structural_reparam()
    
    elif opt.model == 'cnn_swin_fusion':
        model = cnn_swin_fusion(opt)
    
    elif opt.model == 'crop_fusion_att':
        model = crop_fusion_att(opt)

    if opt.load != '':
        print(f'loading pretrained model from {opt.load}')
        model.load_state_dict(torch.load(opt.load))

    if opt.mixup > 0:
        mixup_fn = Mixup(
            mixup_alpha=opt.mixup, cutmix_alpha=0.4, cutmix_minmax=None,
            prob=0.3, switch_prob=0.5, mode='batch',
            label_smoothing=opt.smooth_factor, num_classes=opt.num_classes)
        criterion = SoftTargetCrossEntropy()
    else:
        mixup_fn = None
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=opt.smooth_factor)
    
    if opt.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay = opt.weight_decay)
    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, weight_decay = opt.weight_decay, momentum=0.9, nesterov=True)

    """lr_scheduler"""
    if opt.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=1e-5)
    elif opt.scheduler == 'linearwarmup':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=opt.warmup_epochs, max_epochs=opt.n_epochs, eta_min=1e-6)
    
    main(opt, model, criterion, optimizer, scheduler, train_loader, val_loader, mixup_fn)
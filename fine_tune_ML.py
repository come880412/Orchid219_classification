import numpy as np
import pandas as pd
import copy, os, random, math, tqdm, time
import torch
from PIL import Image
from torch import nn
from torchvision import transforms, models
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import joblib
import timm
import warnings
warnings.filterwarnings("ignore")

size = 384
test_transforms = transforms.Compose([  
                                        transforms.Resize((size,size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.480, 0.423, 0.367], [0.247, 0.241, 0.249])
                                    ])
class OrchidData(Dataset):
    def __init__(self, root, file = 0, mode= 'train'):
        self.transform = test_transforms  
        self.mode = mode
        self.root = root
        self.image_info = pd.read_csv(file).values
        self.image_info = self.image_info[:10]

    def __getitem__(self, index):
        img_path, label = self.image_info[index]
        img = Image.open(os.path.join(self.root,img_path)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.image_info)

def get_acc(y_preds, y_trues):
    total_correct = np.sum(np.equal(y_trues, y_preds))
    total = len(y_trues)
    accuracy = total_correct / total
    return accuracy

def get_wp_f1(y_trues, y_preds, TotalImageCount, num_classes):
    """ Precision_Recall_F1score metrics
    y_pred: the predicted score of each class, shape: (Batch_size, num_classes)
    y_true: the ground truth labels, shape: (Batch_size,) for 'multi-class' or (Batch_size, n_classes) for 'multi-label'
    """
    eps=1e-20
    confusion = confusion_matrix(y_trues, y_preds)

    precision_list = []
    recall_list = []
    F1_dict = {}
    TP_list = []
    FN_list = []
    total_F1 = 0
    for i in range(len(confusion)):
        TP = confusion[i, i]
        FN = sum(confusion[i, :]) - TP
        FP = sum(confusion[:, i]) - TP

        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        F1 = 2 * precision  * recall / (precision + recall + eps)
        total_F1 += F1

        TP_list.append(TP)
        FN_list.append(FN)
        precision_list.append(precision)
        recall_list.append(recall)
        F1_dict[i] = F1
        

    weighted = 0.
    for i in range(len(confusion)):
        weighted += precision_list[i] * (TP_list[i] + FN_list[i])

    WP = weighted / TotalImageCount
    acc = get_acc(y_preds, y_trues)
    Macro_F1 = total_F1 / num_classes
    Final_Score = 0.5*acc + 0.5*Macro_F1
    return acc, Macro_F1, Final_Score, F1_dict

if __name__ in "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 219
    batch_size = 2
    train_data = OrchidData(root = "../dataset/Orchid219/images", file = "../dataset/Orchid219/random_split/train.csv", mode = 'train')
    valid_data = OrchidData(root = "../dataset/Orchid219/images", file = "../dataset/Orchid219/random_split/valid.csv", mode = 'valid')
    test_data = OrchidData(root = "../dataset/Orchid219/images", file = "../dataset/Orchid219/random_split/test.csv", mode = 'test')
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, shuffle = False, num_workers = 0, drop_last=False)
    vaildloader = torch.utils.data.DataLoader(valid_data, batch_size = batch_size, shuffle = False, num_workers = 0, drop_last=False)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, shuffle = False, num_workers = 0, drop_last=False)
    
    print("features.....")
    model = timm.create_model('convnext_xlarge_384_in22ft1k', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load('public_model/convnext_xlarge.pth'))
    model = model.cuda().eval()

    modelswin = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=False, num_classes=num_classes)
    modelswin.load_state_dict(torch.load('public_model/swin_best.pth'))
    modelswin = modelswin.cuda().eval()

    with torch.no_grad():
        for loader , name in [(trainloader, 'train'), (vaildloader, 'valid'), (testloader, 'test')]:
            for idx,(inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(device), labels.to(device)

                logpSwin, _ = modelswin.forward_features(inputs)
                logpSwin = logpSwin.mean(dim=1)

                logps, _ = model.forward_features(inputs)
                logps = model.head.global_pool(logps)
                logps = model.head.norm(logps)
                logps = model.head.flatten(logps)

                logps = torch.cat((logpSwin, logps), dim = 1)

                feature = torch.cat((logps, labels.unsqueeze(1)), dim = 1)
                if idx == 0:
                    features = feature
                else:
                    features = torch.cat((features, feature), dim = 0)
            features = features.cpu().detach().numpy()
            np.savetxt(name + '.csv', features, fmt='%.18e', delimiter = ',')

    print("Train.....")
    train_data = np.loadtxt('train.csv', delimiter=',')
    train_X, train_Y = train_data[:, :-1], train_data[:, -1]
    valid_data = np.loadtxt('valid.csv', delimiter=',')
    valid_X, valid_Y = valid_data[:, :-1], valid_data[:, -1]
    test_data = np.loadtxt('test.csv', delimiter=',')
    test_X, test_Y = test_data[:, :-1], test_data[:, -1]
    train_X = np.concatenate((train_X, valid_X, test_X), axis = 0)
    train_Y = np.concatenate((train_Y, valid_Y, test_Y), axis = 0)

    Logistic = LogisticRegression(penalty = 'none', # 'l1', 'l2', 'elasticnet', 'none'
                                  solver = 'newton-cg', # 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
                                  multi_class = 'auto',  # 'auto', 'ovr', 'multinomial'
                                  max_iter = 100,
                                  random_state = 0,
                                  l1_ratio = 0.5,
                                  n_jobs = -1)

    models = [(Logistic, 'Logistic')]
    for model, model_name in models:
        model.fit(train_X, train_Y)
        joblib.dump(model, model_name + ".sav")

    
    print("Valid and Test .......")
    for model, model_name in models:
        model = joblib.load(model_name + ".sav")
        print('--- %s ---'%(model_name))
        for X, Y, name in [(train_X, train_Y, "Train"), (valid_X, valid_Y, "Valid"), (test_X, test_Y, "Test")]:
            preds = model.predict(X)
            acc, Macro_F1, Final_Score, F1_dict = get_wp_f1(Y, preds, len(Y), num_classes)
            print( "[%s]  Best ACC: %4.5f | Macro F1: %4.5f | Final_Score: %4.5f"%(name, acc, Macro_F1, Final_Score) )

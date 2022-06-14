import numpy as np
import pandas as pd
import copy, os, random, math, tqdm
import torch
from PIL import Image
from torch import nn
from torchvision import transforms, models
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import timm
import warnings
warnings.filterwarnings("ignore")

test_transforms = transforms.Compose([  
                                        transforms.Resize((384,384)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.480, 0.423, 0.367], [0.247, 0.241, 0.249])
                                    ])
test640_transforms = transforms.Compose([  
                                        transforms.Resize((640,640)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.480, 0.423, 0.367], [0.247, 0.241, 0.249])
                                    ])
class OrchidData(Dataset):
    def __init__(self, root, file = 0):
        self.transform = test_transforms  
        self.transform640 = test640_transforms 
        self.root = root
        self.image_info = pd.read_csv(file).values

    def __getitem__(self, index):
        img_name, _ = self.image_info[index]
        img = Image.open(os.path.join(self.root, img_name)).convert("RGB")
        if self.transform is not None:
            img384 = self.transform(img)
            img640 = self.transform640(img)
        return img384, img640, img_name

    def __len__(self):
        return len(self.image_info)

if __name__ in "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 219
    batch_size = 48
    sub_path = "../dataset/Orchid219/submission_template.csv"
    sub_save_path = "../dataset/Swin+Conv_384_to_640+logistic.csv"
    sub_data = OrchidData(root = "../dataset/Orchid219/private_and_public", file = sub_path)
    subloader = torch.utils.data.DataLoader(sub_data, batch_size = batch_size, shuffle = False, num_workers = 4, drop_last=False)
    
    modelswin = timm.create_model('swin_large_patch4_window12_384_in22k', pretrained=False, num_classes=num_classes)
    modelswin.load_state_dict(torch.load('./public_model/swin_best.pth'))
    modelswin = modelswin.cuda().eval()

    model = timm.create_model('convnext_xlarge_384_in22ft1k', pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load('./public_model/convnext_xlarge.pth'))
    model = model.cuda().eval()
    
    Logistic_model = joblib.load("./Logistic.sav")

    sub = pd.read_csv(sub_path).to_dict()
    file_key = { v:k for k, v in sub['filename'].items() }

    with torch.no_grad():
        for idx,(inputs, inputs640, name) in enumerate(tqdm.tqdm(subloader)):
            inputs = inputs.to(device)
            inputs, inputs640 = inputs.to(device), inputs640.to(device)

            logpSwin,_ = modelswin.forward_features(inputs)
            logpSwin = logpSwin.mean(dim=1)

            logps,_ = model.forward_features(inputs640)
            logps = model.head.global_pool(logps)
            logps = model.head.norm(logps)
            logps = model.head.flatten(logps)
            logps = torch.cat((logpSwin, logps), dim = 1)
            logps = logps.cpu().datach().numpy()
            preds = model.predict(logps)

            for ans, name in zip(preds, name):
                sub['category'][file_key[name]] = ans
    df = pd.DataFrame.from_dict(sub)
    df.to_csv(sub_save_path, index = False, header=True)

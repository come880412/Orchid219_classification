import os
import numpy as np
import random
import cv2

np.random.seed(2022)
random.seed(2022)

def train_val_split(root, data_dict, split_ratio):
    os.makedirs(os.path.join(root, "random_split"), exist_ok=True)
    train_data, valid_data, test_data = [["filename,category"]], [["filename,category"]], [["filename,category"]]

    for key, value in data_dict.items():
        label, data = key, value
        data = np.array(data)

        data_len = len(data)

        index = np.random.choice(data_len, data_len, replace=False)
        train_index = index[:int(data_len * split_ratio[0])]
        valid_index = index[int(data_len * split_ratio[0]): int(data_len * (split_ratio[0] + split_ratio[1]))]
        test_index = index[int(data_len * (split_ratio[0] + split_ratio[1])):]

        for idx in train_index:
            temp_data = [f"{data[idx]},{label}" ]
            train_data.append(temp_data)
        
        for idx in valid_index:
            temp_data = [f"{data[idx]},{label}" ]
            valid_data.append(temp_data)
        
        for idx in test_index:
            temp_data = [f"{data[idx]},{label}" ]
            test_data.append(temp_data)
    
    np.savetxt(os.path.join(root, "random_split", 'train.csv'),  train_data, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(root, "random_split", 'valid.csv'),  valid_data, fmt='%s', delimiter=',')
    np.savetxt(os.path.join(root, "random_split", 'test.csv'),  test_data, fmt='%s', delimiter=',')
    return [len(train_data), len(valid_data), len(test_data)]

def cross_validation(root, data_dict, k_fold):

    for k in range(0, k_fold):
        os.makedirs(os.path.join(root, 'k_fold', str(k+1)), exist_ok=True)
        train_data, valid_data = [["filename,category"]], [["filename,category"]]
        for key, value in data_dict.items():
            label, data = key, value

            data_len = len(data)
            ratio = data_len // k_fold
            index = [i for i in range(data_len)]

            train_index = index[:(k * ratio)] + index[((k+1) * ratio):]
            valid_index = index[k*ratio : (k+1) * ratio]

            for idx in train_index:
                temp_data = [f"{data[idx]},{label}" ]
                train_data.append(temp_data)
            
            for idx in valid_index:
                temp_data = [f"{data[idx]},{label}" ]
                valid_data.append(temp_data)
        
        np.savetxt(os.path.join(root, 'k_fold', str(k+1), 'train.csv'),  train_data, fmt='%s', delimiter=',')
        np.savetxt(os.path.join(root, 'k_fold', str(k+1), 'valid.csv'),  valid_data, fmt='%s', delimiter=',')

def Norm(train_path):
    # img_h, img_w = 32, 32
    img_h, img_w = 384, 384   #根据自己数据集适当调整，影响不大
    means, stdevs = [], []
    img_list = []
    
    train_path = os.path.join(train_path, 'images')
    train_paths = os.listdir(train_path)
    imgs_path_list = []
    for image_name in train_paths:
        imgs_path_list.append(os.path.join(os.path.join(train_path, image_name)))
    
    len_ = len(imgs_path_list)
    i = 0
    for path in imgs_path_list:
        img = cv2.imread(path)
        img = cv2.resize(img, (img_w, img_h))
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
        i += 1
        if i % 1000 == 0:
            print(i,'/',len_)    
    
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255.
    
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    
    # BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换
    means.reverse()
    stdevs.reverse()
    return means, stdevs

if __name__ == '__main__':
    root = '../dataset/Orchid219'
    split_ratio = [0.7,0.2,0.1] # train/valid/test
    k_fold = 5

    data_dict = {}

    infos = np.loadtxt(os.path.join(root, 'label.csv'), delimiter=',', dtype=np.str)[1:]

    for info in infos:
        file_name, label = info

        if label not in data_dict.keys():
            data_dict[label] = []
        
        data_dict[label].append(file_name)
    
    mean, std = Norm(root)

    len_data = train_val_split(root, data_dict, split_ratio)
    # cross_validation(root, data_dict, k_fold)

    print('------------Statistics------------')
    print('Number of training data: ', len_data[0])
    print('Number of validation data: ', len_data[1])
    print('Number of testing data: ', len_data[2])
    print('Mean of dataset: ', mean)
    print('Std of dataset', std)
         

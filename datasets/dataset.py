import os
import json
import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .augmentations import *

class ShapenetDataset(Dataset):

    def __init__(self, root, split, gt_npoints=4096, 
                 in_npoints = 1024, class_choice=None):

        self.root = root
        self.split = split.lower()
        self.gt_npoints = gt_npoints
        self.in_npoints = in_npoints
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}

        # Open the Category File and Map Folders to Categories
        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        
        # select specific categories from the dataset. 
        # ex: Call in parameters "class_choice=["Airplane"].
        if not class_choice is  None:
            self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
        
        # get train, valid, test splits from json files
        if self.split == 'train':
            split_file = os.path.join(self.root, 
                r'train_test_split/shuffled_train_file_list.json')
        elif self.split == 'test':
            split_file = os.path.join(self.root, 
                r'train_test_split/shuffled_test_file_list.json')
        elif (self.split == 'valid') or (self.split == 'val'):
            split_file = os.path.join(self.root, 
                r'train_test_split/shuffled_val_file_list.json')
        
        # for each category, assign the point, segmentation, and image.
        self.meta = {}        
        for item in self.cat:
            self.meta[item] = []
            dir_point = os.path.join(self.root, self.cat[item], 'points')
            dir_seg = os.path.join(self.root, self.cat[item], 'points_label')
            dir_seg_img = os.path.join(self.root, self.cat[item], 'seg_img')
                
            with open(split_file, 'r') as f:
                split_data = json.load(f)

            # get point cloud file (.pts) names for current split
            pts_names = []
            for token in split_data:
                if self.cat[item] in token:
                    pts_names.append(token.split('/')[-1] + '.pts')

            # FOR EVERY POINT CLOUD FILE
            for fn in pts_names:
                token = (os.path.splitext(os.path.basename(fn))[0])
                # add point cloud, segmentations, and image to class metadata dict
                self.meta[item].append((os.path.join(dir_point, token + '.pts'), 
                                        os.path.join(dir_seg, token + '.seg'),
                                        os.path.join(dir_seg_img, token + '.png')))
       
        # create list containing (item, points, segmentation points, segmentation image)
        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1], fn[2]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        
        self.num_seg_classes = 0
        for i in range(len(self.datapath)//50):
            # get number of seg classes in current item
            l = len(np.unique(np.loadtxt(self.datapath[i][2]).astype(np.uint8)))
            if l > self.num_seg_classes:
                self.num_seg_classes = l
        
        # Augmentation
        if self.split == 'train':
            self.transform = transforms.Compose([
                # Rotate(rotate_y=True),
                Normalize(),
                PartialView(radius=1.4, r_scale=40),
                SamplePoints(self.in_npoints, self.gt_npoints),
                # AddNoise(),
                # AddRandomPoints(0.02),
            ])
        else: # test
            self.transform = transforms.Compose([
                Normalize(),
                PartialView(radius=1.4, r_scale=40,random=False),
                SamplePoints(self.in_npoints, self.gt_npoints),
            ])


    def __getitem__(self, index):
        '''
        Each element has the format "class, points, segmentation labels, segmentation image"
        '''
        # Get one Element
        fn = self.datapath[index]
        
        # get its Class
        cls_ = self.classes[fn[0]]
        
        # Read the Point Cloud
        point_set = np.asarray(o3d.io.read_point_cloud(fn[1], format='xyz').points,dtype=np.float32)
        
        # Read the Segmentation Data
        seg = np.loadtxt(fn[2]).astype(np.int64)
                                
        # Transform
        sample = {"in_points": point_set, "gt_points": point_set, "seg": seg}
        return self.transform(sample)

    def __len__(self):
        return len(self.datapath)


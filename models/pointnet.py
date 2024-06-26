import torch
import torch.nn as nn
import torch.nn.functional as F

class Tnet(nn.Module):
    ''' Learns a transformation matrix with the specified dimensions '''
    def __init__(self, dim, num_points=2500) -> None:
        super().__init__()

        self.dim = dim

        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, dim**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points)
        
    def forward(self, x: torch.Tensor):
        bs = x.shape[0]

        x = self.bn1(F.relu((self.conv1(x))))
        x = self.bn2(F.relu((self.conv2(x))))
        x = self.bn3(F.relu((self.conv3(x))))

        x = self.max_pool(x).view(bs, -1)

        x = self.bn4(F.relu(self.linear1(x)))
        x = self.bn5(F.relu(self.linear2(x)))
        x = self.linear3(x)

        iden = torch.eye(self.dim, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda():
            iden = iden.cuda()
        
        x = x.view(-1, self.dim, self.dim) + iden

        return x

class PointNetBackbone(nn.Module):
    def __init__(self, num_points=2500, num_global_feats=1024, local_feat=True) -> None:
        super().__init__()
        self.num_global_feats = num_global_feats
        self.num_points = num_points
        self.local_feat = local_feat

        self.t_net1 = Tnet(3, num_points=num_points)
        self.t_net2 = Tnet(64, num_points=num_points)

        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv5 = nn.Conv1d(128, num_global_feats, kernel_size=1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.num_global_feats)

        self.max_pool = nn.MaxPool1d(kernel_size=num_points, return_indices=True)
    
    def forward(self, x: torch.Tensor):
        bs = x.shape[0]

        t_input = self.t_net1(x)
        x = torch.bmm(x.transpose(2, 1), t_input).transpose(2, 1)

        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))

        t_feat = self.t_net2(x)
        x = torch.bmm(x.transpose(2, 1), t_feat).transpose(2, 1)

        # Store local features
        local_feats = x.clone()

        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))

        global_features, critical_indexes = self.max_pool(x)
        global_features = global_features.view(bs, -1)
        critical_indexes = critical_indexes.view(bs, -1)

        if self.local_feat:
            features = torch.cat((local_feats,
                                 global_features.unsqueeze(-1).repeat(1, 1, self.num_points)),
                                 dim=1)
            return features, critical_indexes, t_feat
        else:
            return global_features, critical_indexes, t_feat

class PointNetClassHead(nn.Module):
    '''' Classification Head '''
    def __init__(self, num_points=2500, num_global_feats=1024, k=2):
        super(PointNetClassHead, self).__init__()

        # get the backbone (only need global features for classification)
        self.backbone = PointNetBackbone(num_points, num_global_feats, local_feat=False)

        # MLP for classification
        self.linear1 = nn.Linear(num_global_feats, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, k)

        # batchnorm for the first 2 linear layers
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)

        # The paper states that batch norm was only added to the layer 
        # before the classication layer, but another version adds dropout  
        # to the first 2 layers
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        # get global features
        x, crit_idxs, A_feat = self.backbone(x) 

        x = self.bn1(F.relu(self.linear1(x)))
        x = self.bn2(F.relu(self.linear2(x)))
        x = self.dropout(x)
        x = self.linear3(x)

        # return logits
        return x, crit_idxs, A_feat


class PointNetSegHead(nn.Module):
    ''' Segmentation Head '''
    def __init__(self, num_points=2500, num_global_feats=1024, m=2):
        super(PointNetSegHead, self).__init__()

        self.num_points = num_points
        self.m = m

        # get the backbone 
        self.backbone = PointNetBackbone(num_points, num_global_feats, local_feat=True)

        # shared MLP
        num_features = num_global_feats + 64 # local and global features
        self.conv1 = nn.Conv1d(num_features, 512, kernel_size=1)
        self.conv2 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=1)
        self.conv4 = nn.Conv1d(128, m, kernel_size=1)

        # batch norms for shared MLP
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)


    def forward(self, x):
        
        # get combined features
        x, crit_idxs, A_feat = self.backbone(x) 

        # pass through shared MLP
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.conv4(x)

        x = x.transpose(2, 1)
        
        return x, crit_idxs, A_feat


a = torch.ones((3, 1))
test = Tnet(3)
test(a)

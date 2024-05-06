import torch
import torch.backends
import torch.backends.cudnn
import torch.utils
import torch.utils.data

from datasets.dataset import ShapenetDataset
from models.PCN import PCN
from models.PoinTr import PoinTr
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2

import numpy as np
import time, random
import open3d as o3d

def train(cfgs):
    # Set-up device
    if cfgs["device"] == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = cfgs["device"]

    # Set random seed
    if cfgs["random_seed"]:
        torch.manual_seed(cfgs["random_seed"])
        random.seed(cfgs["random_seed"])
        np.random.seed(0)

    # Dataset
    data_cfg = cfgs["dataset"]
    train_dataset = ShapenetDataset(data_cfg["path"], split='train',
                                    in_npoints=data_cfg["in_points"], 
                                    class_choice=data_cfg["class_choice"])
    train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                              batch_size=data_cfg["batch_size"],
                                              shuffle=True)
    val_dataset = ShapenetDataset('data/ShapeNetPart_mini/', split='train',
                                     in_npoints=data_cfg["in_points"])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                              batch_size=data_cfg["batch_size"], 
                                              shuffle=False)
    
    # Create model
    # TODO: Set different models
    model = PCN().to(device)

    # parameter setting
    start_epoch = 0 
    best_metrics = None
    metrics = None

    # TODO: resume checkpoint

    # Print model info
    print("Using model:\n", model)

    # optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=cfgs["optim"]["lr"], 
                                weight_decay=cfgs["optim"]["weight_decay"])
    # # TODO: resume(?) scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=cfgs["scheduler"]["step_size"],
                                                gamma=cfgs["scheduler"]["gamma"])
    
    # Criterion
    ChamferDisL1 = ChamferDistanceL1()
    ChamferDisL2 = ChamferDistanceL2()

    # Training
    model.zero_grad()
    for epoch in range(start_epoch, start_epoch + cfgs["num_epochs"]):
        # Meters
        epoch_start_time = time.time()
        batch_start_time = time.time()
        # batch_time = AverageMeter()
        # data_time = AverageMeter()
        # losses = AverageMeter(['SparseLoss', 'DenseLoss'])

        # Set model to training mode
        model.train()
        n_batches = len(train_dataloader)
        for idx, data in enumerate(train_dataloader):
            points, gt, segs = data["in_points"], data['gt_points'], data["seg"]

            # Transform data to float and move to device
            points: torch.Tensor = points.float().to(device)
            gt: torch.Tensor = gt.float().to(device)
            segs: torch.Tensor = segs.float().to(device)
            
            # Forward
            pred = model(points)
            
            # Loss
            sparse_loss, dense_loss = model.get_loss(pred, gt, epoch)
            loss = sparse_loss + dense_loss
            loss.backward()

            # Optimize
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                           10, norm_type=2)
            optimizer.step()
            model.zero_grad()

            # Log
            print(epoch, sparse_loss, dense_loss)
    
        # Scheduler
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()

        
    for data in val_dataloader:
        model.eval()
        points, gt, segs = data["in_points"], data['gt_points'], data["seg"]
        points: torch.Tensor = points.float().to(device)

        pred = model(points)

        in_cloud = points.cpu().detach().numpy()
        coarse = pred[0].cpu().detach().numpy()
        fine = pred[1].cpu().detach().numpy()

        in_pcd = o3d.geometry.PointCloud()
        in_pcd.points = o3d.utility.Vector3dVector(in_cloud[0])
        fine_pcd = o3d.geometry.PointCloud()
        fine_pcd.points = o3d.utility.Vector3dVector(fine[0])
        coarse_pcd = o3d.geometry.PointCloud()
        coarse_pcd.points = o3d.utility.Vector3dVector(coarse[0])
        # o3d.visualization.draw_geometries([pcd])
        # o3d.visualization.draw_plotly([in_pcd])

        o3d.io.write_point_cloud("tests/cloud_in.ply", in_pcd)
        o3d.io.write_point_cloud("tests/cloud_fine.ply", fine_pcd)
        o3d.io.write_point_cloud("tests/cloud_coarse.ply", coarse_pcd)
        break



def main():
    # TODO: Define parameters
    cfgs = {
        "num_epochs": 10,
        "device": 'cuda',
        "random_seed": 0,
        "dataset": {
            "path": 'data/shapenetcore_partanno_segmentation_benchmark_v0/',
            "class_choice": None,
            "in_points": 2048,
            "batch_size": 32
        },
        "optimizer": {
            "lr": 0.0001, 
            "weight_decay": 0
        },
        "scheduler": {
            "step_size": 50,
            "gamma": 0.5
        }
    }

    # Call training function
    train(cfgs)
    

if __name__=="__main__":
    main()
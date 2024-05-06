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
import os, datetime
import json


def train(cfgs):
    # Set-up device
    if cfgs["device"] == 'cuda' and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        device = cfgs["device"]
    else:
        device = 'cpu'

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
    val_dataset = ShapenetDataset(data_cfg["path"], split='val',
                                     in_npoints=data_cfg["in_points"],
                                     class_choice=data_cfg["class_choice"])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                              batch_size=1, shuffle=False)
    
    # Create model
    # TODO: Set different models
    model = PoinTr().to(device)

    # parameter setting
    start_epoch = 0 
    best_metric = None

    # Create folder
    experiment_path = os.path.join('checkpoints/', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(experiment_path)
    with open(os.path.join(experiment_path, 'cfgs.json'), 'w') as outfile:
        json.dump(cfgs, outfile)

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
            print(epoch, loss)
    
        # Scheduler
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step()
        else:
            scheduler.step()

        # Validate
        if epoch % cfgs['val_freq'] == 0 or epoch == start_epoch + cfgs["num_epochs"] - 1:
            metric = validate(model, val_dataloader, epoch, device, ChamferDisL1, ChamferDisL2)
            
            if best_metric == None or metric < best_metric:
                save_checkpoint(model, experiment_path, 'ckpt-best.pth')
                with open(os.path.join(experiment_path, 'metrics.txt'), 'a') as outfile:
                    outfile.write(f'{epoch}: {metric}\n')
        # Always save last checkpoint
        save_checkpoint(model, experiment_path, 'ckpt-last.pth')



def validate(model, val_dataloader, epoch, device, ChamferDisL1, ChamferDisL2):
    print(f"[VALIDATION] Start validation for epoch {epoch}")
    model.eval()

    loss_sum = 0
    count = 0
    with torch.no_grad():
        for data in val_dataloader:
            points, gt, segs = data["in_points"], data['gt_points'], data["seg"]
            points: torch.Tensor = points.float().to(device)
            gt: torch.Tensor = gt.float().to(device)

            pred = model(points)
            coarse_points = pred[0]
            dense_points = pred[1]

            sparse_loss_l1 =  ChamferDisL1(coarse_points, gt)
            sparse_loss_l2 =  ChamferDisL2(coarse_points, gt)
            dense_loss_l1 =  ChamferDisL1(dense_points, gt)
            dense_loss_l2 =  ChamferDisL2(dense_points, gt)

            loss_sum += sparse_loss_l1 + sparse_loss_l2 + dense_loss_l1 + dense_loss_l2
            count += 1

        metric = loss_sum * 1000 / count
        print(f"model had a end metric of {metric}")
        return metric


def save_checkpoint(model, experiment_path, name):
    torch.save(model.state_dict(), os.path.join(experiment_path, name))


def main():
    # TODO: Define parameters
    cfgs = {
        "num_epochs": 100,
        "device": 'cuda',
        "random_seed": 0,
        'val_freq': 10,
        "dataset": {
            "path": 'data/shapenetcore_partanno_segmentation_benchmark_v0/',
            "class_choice": "Airplane",
            "in_points": 2048,
            "batch_size": 32
        },
        "optim": {
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
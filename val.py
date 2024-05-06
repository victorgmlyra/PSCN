import torch
import torch.backends
import torch.backends.cudnn
import torch.utils
import torch.utils.data

from datasets.dataset import ShapenetDataset
from models.PCN import PCN
from models.PoinTr import PoinTr

import os
import json
import open3d as o3d


ckpt_path = os.path.join('checkpoints', '2024-05-06_16-12-40')
with open(os.path.join(ckpt_path, 'cfgs.json'), 'r') as f:
    cfg = json.load(f)


data_cfg = cfg['dataset']
val_dataset = ShapenetDataset(data_cfg["path"], split='val',
                            in_npoints=data_cfg["in_points"],
                            class_choice=data_cfg["class_choice"])
val_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                    batch_size=10, shuffle=False)
                                             

device = cfg['device']
model = PoinTr().to(device)
model.load_state_dict(torch.load(os.path.join(ckpt_path, 'ckpt-best.pth')))
model.eval()


for data in val_dataloader:
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
    # o3d.visualization.draw_geometries([fine_pcd])
    # o3d.visualization.draw_plotly([in_pcd])

    o3d.io.write_point_cloud("tests/cloud_in.ply", in_pcd)
    o3d.io.write_point_cloud("tests/cloud_fine.ply", fine_pcd)
    o3d.io.write_point_cloud("tests/cloud_coarse.ply", coarse_pcd)
    break

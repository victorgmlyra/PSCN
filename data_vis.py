import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader

import open3d as o3d
import numpy as np

from datasets.dataset import ShapenetDataset

def read_pointnet_colors(seg_labels):
    ''' map segmentation labels to colors '''
    map_label_to_rgb = {
        1: [0, 255, 0],
        2: [0, 0, 255],
        3: [255, 0, 0],
        4: [255, 0, 255],  # purple
        5: [0, 255, 255],  # cyan
        6: [255, 255, 0],  # yellow
    }
    colors = np.array([map_label_to_rgb[label] for label in seg_labels])
    return colors


sample_dataset = ShapenetDataset('data/ShapeNetPart_mini/', 
                                 split='train')


data_loader = torch.utils.data.DataLoader(sample_dataset, batch_size=1, shuffle=True)
for data in data_loader:
    points, gt, segs = data["in_points"], data['gt_points'], data["seg"]
    print(points.shape, gt.shape)
    points: torch.Tensor

    in_pcd = o3d.geometry.PointCloud()
    in_pcd.points = o3d.utility.Vector3dVector(points.numpy()[0])
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt.numpy()[0])
    gt_pcd.colors = o3d.utility.Vector3dVector(read_pointnet_colors(segs.numpy()[0]))
    # o3d.visualization.draw_geometries([pcd])
    # o3d.visualization.draw_plotly([in_pcd])

    o3d.io.write_point_cloud("input.ply", in_pcd)
    o3d.io.write_point_cloud("gt.ply", gt_pcd)

    break

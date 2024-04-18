import numpy as np
import open3d as o3d


def read_pointnet_colors(seg_labels):
    ''' map segementation labels to colors '''
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


def find_bounding_circle(pcd):
    ''' Get an approximate bounding circle of the object. '''
    bb: o3d.geometry.AxisAlignedBoundingBox = pcd.get_axis_aligned_bounding_box()
    center_point = (bb.min_bound + bb.max_bound) / 2
    approx_radius = np.max(np.abs(np.hstack([center_point - bb.min_bound, center_point - bb.max_bound])))
    approx_radius *= 1.3
    return center_point, approx_radius


def random_point_on_sphere(center, radius):
    phi = np.random.uniform(0, 2 * np.pi)
    theta = np.random.uniform(0, np.pi)

    # Spherical to Cartesian conversion
    x = center[0] + radius * np.sin(theta) * np.cos(phi)
    y = center[1] + radius * np.sin(theta) * np.sin(phi)
    z = center[2] + radius * np.cos(theta)

    return np.array([[x, y, z]])


def get_visible_points(pcd, cam):
    visible_points = [cam[0]]
    for point, normal in zip(np.asarray(pcd.points), np.asarray(pcd.normals)):
        vec_to_cam = cam - point

        angle = np.arccos(np.clip(np.dot(vec_to_cam, normal), -1.0, 1.0))

        if angle < np.pi/2:
            visible_points.append(point)
        
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(visible_points))

    return pcd


data_folder = 'data/ShapeNetPart_mini/02691156'
points_file = data_folder + '/points/1a32f10b20170883663e90eaf6b4ca52.pts'
label_file = data_folder + '/points_label/1a32f10b20170883663e90eaf6b4ca52.seg'

points = np.asarray(o3d.io.read_point_cloud(points_file, format='xyz').points, dtype=np.float32)
segs = np.loadtxt(label_file).astype(np.int64)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(read_pointnet_colors(segs))

c, r = find_bounding_circle(pcd)
random_point = random_point_on_sphere(c, r)

# partial_pcd = get_visible_points(pcd, random_point)
_, pt_map = pcd.hidden_point_removal(random_point[0], r*50)
partial_pcd = pcd.select_by_index(pt_map)

# partial_pcd.points.extend(o3d.utility.Vector3dVector(random_point))
#(o3d.utility.Vector3dVector(random_point)))

o3d.visualization.draw_geometries([partial_pcd], point_show_normal=True)

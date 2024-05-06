import numpy as np
import open3d as o3d


class Normalize(object):
    """ Perform min/max normalization on points """
    def __call__(self, sample):
        points, seg = sample['gt_points'], sample['seg']

        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.sqrt(np.max(np.sum(points**2,axis=-1)))
        points /= furthest_distance

        return {"in_points": points, "gt_points": points, "seg": seg}


class SamplePoints(object):
    """ Sample input and gt point clouds """
    def __init__(self, in_npoints=1024, gt_npoints=4096) -> None:
        self.gt_npoints = gt_npoints
        self.in_npoints = in_npoints

    def __call__(self, sample):
        in_points, gt_points, seg = sample['in_points'], sample['gt_points'], sample['seg']

        # Sample ground truth points and segmentation labels
        gt_points, seg = self._sample_points(gt_points, seg, self.gt_npoints)

        # Sample input points
        in_points, _ = self._sample_points(in_points, None, self.in_npoints)

        return {"in_points": in_points, "gt_points": gt_points, "seg": seg}
    
    def _sample_points(self, points, labels, npoints):
        """Helper function to sample points and corresponding labels if provided. """
        num_points = len(points)
        if num_points >= npoints:
            choice = np.random.choice(num_points, npoints, replace=False)
        else:
            choice = np.random.choice(num_points, npoints, replace=True)
        
        sampled_points = points[choice, :]
        if labels is not None:
            sampled_labels = labels[choice]
            return sampled_points, sampled_labels
        return sampled_points, None


class Rotate(object):
    """ Randomly rotates point cloud about specified axes.
    
    Args:
        rotate_x (bool): If True, rotate about the x-axis.
        rotate_y (bool): If True, rotate about the y-axis.
        rotate_z (bool): If True, rotate about the z-axis.
    """
    def __init__(self, rotate_x=False, rotate_y=False, rotate_z=False):
        self.rotate_x = rotate_x
        self.rotate_y = rotate_y
        self.rotate_z = rotate_z

    def __call__(self, sample):
        points, seg = sample['gt_points'], sample['seg']

        # Random angle generation for each axis
        phi = np.random.uniform(-np.pi, np.pi) if self.rotate_x else 0
        theta = np.random.uniform(-np.pi, np.pi) if self.rotate_y else 0
        psi = np.random.uniform(-np.pi, np.pi) if self.rotate_z else 0

        # Rotation matrices for each axis
        rot_x = np.array([
            [1,            0,           0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])

        rot_y = np.array([
            [np.cos(theta),  0, np.sin(theta)],
            [0,               1,            0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        rot_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi),  np.cos(psi),  0],
            [0,            0,            1]
        ])

        # Composite rotation based on enabled axes
        rot = np.dot(rot_x, np.dot(rot_y, rot_z))

        # Rotate points
        points = np.dot(points, rot)

        return {"in_points": points, "gt_points": points, "seg": seg}

class PartialView(object):
    """ Gets random partial view of the object """
    def __init__(self, radius=1, r_scale= 50, random=True):
        self.r = radius
        self.r_scale = r_scale
        self.random = random

    def __call__(self, sample):
        points, seg = sample['gt_points'], sample['seg']

        in_points = np.copy(points)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(in_points)
        random_point = self.random_point_on_unit_sphere()
        _, pt_map = pcd.hidden_point_removal(random_point, self.r*self.r_scale)
        partial_pcd = pcd.select_by_index(pt_map)

        return {"in_points": np.asarray(partial_pcd.points), "gt_points": points, "seg": seg}
    
    def random_point_on_unit_sphere(self):
        if self.random:
            phi = np.random.uniform(0, 2 * np.pi)
            theta = np.random.uniform(0, np.pi)
        else:
            phi = np.pi/4
            theta = np.pi/4

        # Spherical to Cartesian conversion
        x = self.r * np.sin(theta) * np.cos(phi)
        y = self.r * np.sin(theta) * np.sin(phi)
        z = self.r * np.cos(theta)

        return np.array([x, y, z])
    
class AddNoise(object):
    """ Add gaussian noise """
    def __call__(self, sample):
        points, gt, seg = sample["in_points"], sample['gt_points'], sample["seg"]

        # add N(0, 1/100) noise
        points += np.random.randn(*points.shape) / 100

        return {"in_points": points, "gt_points": gt, "seg": seg}


class AddRandomPoints(object):
    """ Add random points given a probability """
    def __init__(self, prob=0.05) -> None:
        self.prob = prob

    def __call__(self, sample):
        points, gt, seg = sample["in_points"], sample['gt_points'], sample["seg"]

        # Generate random boolean mask based on probability
        num_points = points.shape[0]
        mask = np.random.rand(num_points) < self.prob

        # Generate random points within the range [-1, 1] for the points to be replaced
        random_points = np.random.normal(0, 0.2, (num_points, 3))  
        random_points = np.clip(random_points, -1, 1) 

        # Substitute points where mask is True
        points[mask] = random_points[mask]

        return {"in_points": points, "gt_points": gt, "seg": seg}
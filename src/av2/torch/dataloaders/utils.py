"""Pytorch detection dataloader utilities."""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from enum import Enum, unique
from functools import cached_property
from typing import Final, Optional, Tuple, List
from av2.utils.typing import NDArrayByte, NDArrayFloat, NDArrayInt

import fsspec.asyn
import numpy as np
import pandas as pd
import torch
from torch import Tensor

import av2._r as rust
from av2.geometry.geometry import mat_to_xyz, quat_to_mat
from av2.geometry.se3 import SE3
from av2.utils.typing import NDArrayFloat
from av2.structures.cuboid import CuboidList



CATEGORY_MAP = {"ANIMAL":0, "ARTICULATED_BUS":1, "BICYCLE":2, "BICYCLIST":3, "BOLLARD":4,
                "BOX_TRUCK":5, "BUS":6, "CONSTRUCTION_BARREL":7, "CONSTRUCTION_CONE":8, "DOG":9,
                "LARGE_VEHICLE":10, "MESSAGE_BOARD_TRAILER":11, "MOBILE_PEDESTRIAN_CROSSING_SIGN":12,
                "MOTORCYCLE":13, "MOTORCYCLIST":14, "OFFICIAL_SIGNALER":15, "PEDESTRIAN":16,
                "RAILED_VEHICLE":17, "REGULAR_VEHICLE":18, "SCHOOL_BUS":19, "SIGN":20,
                "STOP_SIGN":21, "STROLLER":22, "TRAFFIC_LIGHT_TRAILER":23, "TRUCK":24,
                "TRUCK_CAB":25, "VEHICULAR_TRAILER":26, "WHEELCHAIR":27, "WHEELED_DEVICE":28,
                "WHEELED_RIDER":29}


MAX_STR_LEN: Final[int] = 32

DEFAULT_ANNOTATIONS_TENSOR_FIELDS: Final = [
    "tx_m",
    "ty_m",
    "tz_m",
    "length_m",
    "width_m",
    "height_m",
    "qw",
    "qx",
    "qy",
    "qz",
]
DEFAULT_LIDAR_TENSOR_FIELDS: Final = ["x", "y", "z"]
QUAT_WXYZ_FIELDS: Final = ["qw", "qx", "qy", "qz"]
TRANSLATION_FIELDS: Final = ["tx_m", "ty_m", "tz_m"]


@unique
class OrientationMode(str, Enum):
    """Orientation (pose) modes for the ground truth annotations."""

    QUATERNION_WXYZ = "QUATERNION_WXYZ"
    YAW = "YAW"


@unique
class CuboidMode(str, Enum):
    """Box mode (parameterization) of ground truth annotations."""

    XYZLWH_THETA = "XYZLWH_THETA"
    XYZLWH_QWXYZ = "XYZLWH_QWXYZ"
    XYZ = "XYZ"

    @staticmethod
    def convert(dataframe: pd.DataFrame, src: CuboidMode, target: CuboidMode) -> pd.DataFrame:
        """Convert an annotations dataframe from src to target cuboid parameterization.

        Args:
            dataframe: Annotations dataframe.
            src: Cuboid parameterization of the dataframe.
            target: Desired parameterization of the dataframe.

        Returns:
            The dataframe in the new parameterization format.

        Raises:
            NotImplementedError: If the cuboid mode conversion isn't supported.
        """
        if src == target:
            return dataframe
        if src == CuboidMode.XYZLWH_QWXYZ and target == CuboidMode.XYZLWH_THETA:
            quaternions = dataframe.loc[:, list(QUAT_WXYZ_FIELDS)].to_numpy().astype(np.float64)
            rotation = quat_to_mat(quaternions)
            yaw = mat_to_xyz(rotation)[:, -1]

            first_occurence = min(
                i if field_name in QUAT_WXYZ_FIELDS else sys.maxsize
                for (i, field_name) in enumerate(DEFAULT_ANNOTATIONS_TENSOR_FIELDS)
            )
            field_ordering = tuple(
                filter(lambda field_name: field_name not in QUAT_WXYZ_FIELDS, DEFAULT_ANNOTATIONS_TENSOR_FIELDS)
            )
            field_ordering = field_ordering[:first_occurence] + ("yaw",) + field_ordering[first_occurence:]
            dataframe["yaw"] = yaw
            dataframe = dataframe[list(field_ordering)]
        elif src == CuboidMode.XYZLWH_QWXYZ and target == CuboidMode.XYZ:
            unit_vertices_obj_xyz_m = np.array(
                [
                    [+1, +1, +1],  # 0
                    [+1, -1, +1],  # 1
                    [+1, -1, -1],  # 2
                    [+1, +1, -1],  # 3
                    [-1, +1, +1],  # 4
                    [-1, -1, +1],  # 5
                    [-1, -1, -1],  # 6
                    [-1, +1, -1],  # 7
                ],
            )

            dims_lwh_m = dataframe.loc[:, ["length_m", "width_m", "height_m"]].to_numpy().astype(np.float64)

            # Transform unit polygons.
            vertices_obj_xyz_m = (dims_lwh_m[:, None] / 2.0) * unit_vertices_obj_xyz_m[None]

            quat = dataframe.loc[:, list(QUAT_WXYZ_FIELDS)].to_numpy().astype(np.float64)
            rotation = quat_to_mat(quat)
            translation = dataframe.loc[:, ["tx_m", "ty_m", "tz_m"]].to_numpy().astype(np.float64)

            vertices = (rotation @ vertices_obj_xyz_m.transpose(0, 2, 1)).transpose(0, 2, 1) + translation[:, None]
            columns = list(
                itertools.chain.from_iterable(
                    [(f"tx_{i}", f"ty_{i}", f"tz_{i}") for i in range(len(unit_vertices_obj_xyz_m))]
                )
            )
            vertices = vertices.reshape(-1, len(unit_vertices_obj_xyz_m) * 3)
            dataframe = dataframe.loc[:, ["tx_m", "ty_m", "tz_m", "qw", "qx", "qy", "qz"]]
            dataframe[:, columns] = vertices
            return dataframe
        else:
            raise NotImplementedError("This conversion is not implemented!")
        return dataframe


@dataclass(frozen=True)
class Annotations:
    """Dataclass for ground truth annotations.

    Args:
        dataframe: Dataframe containing the annotations and their attributes.
        cuboid_mode: Cuboid parameterization mode.
    """

    dataframe: pd.DataFrame
    cuboid_mode: CuboidMode = CuboidMode.XYZLWH_QWXYZ

    @property
    def category_names(self) -> List[str]:
        """Return the category names."""
        category_names: List[str] = self.dataframe["category"].to_list()
        return category_names

    @property
    def track_uuids(self) -> List[str]:
        """Return the unique track identifiers."""
        category_names: List[str] = self.dataframe["track_uuid"].to_list()
        return category_names

    def as_tensor(
        self,
        cuboid_mode: CuboidMode = CuboidMode.XYZLWH_THETA,
        dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """Return the annotations as a tensor.

        Args:
            cuboid_mode: Target parameterization for the cuboids.
            dtype: Target datatype for casting.

        Returns:
            (N,K) tensor where N is the number of annotations and K
                is the number of annotation fields.
        """
        dataframe = CuboidMode.convert(self.dataframe, self.cuboid_mode, cuboid_mode)
        return torch.as_tensor(dataframe.to_numpy(), dtype=dtype)

    def compute_interior_points(self, lidar: Lidar) -> Tensor:
        """Compute a pairwise interior point mask.

        Args:
            lidar: Lidar object.

        Returns:
            (num_annotations,num_points) boolean tensor indicating whether the point
                falls into the kth cuboid.
        """
        dataframe = CuboidMode.convert(self.dataframe, self.cuboid_mode, CuboidMode.XYZ)
        points_xyz = lidar.as_tensor()

        columns = list(itertools.chain.from_iterable([(f"tx_{i}", f"ty_{i}", f"tz_{i}") for i in range(8)]))
        cuboid_vertices = torch.as_tensor(dataframe[columns].to_numpy(), dtype=torch.float32).reshape(-1, 8, 3)
        pairwise_point_masks = compute_interior_points_mask(points_xyz, cuboid_vertices)
        return pairwise_point_masks


@dataclass(frozen=True)
class Lidar:
    """Dataclass for lidar sweeps.

    Args:
        dataframe: Dataframe containing the lidar and its attributes.
    """

    dataframe: pd.DataFrame

    def as_tensor(
        self, field_ordering: Tuple[str, ...] = DEFAULT_LIDAR_TENSOR_FIELDS, dtype: torch.dtype = torch.float32
    ) -> Tensor:
        """Return the lidar sweep as a dense tensor.

        Args:
            field_ordering: Feature ordering for the tensor.
            dtype: Target datatype for casting.

        Returns:
            (N,K) tensor where N is the number of lidar points and K
                is the number of features.
        """
        dataframe_npy = self.dataframe.loc[:, list(field_ordering)].to_numpy()
        return torch.as_tensor(dataframe_npy, dtype=dtype)


@dataclass(frozen=True)
class Sweep:
    """Stores the annotations and lidar for one sweep.

    Args:
        annotations: Object containing annotation parameters.
        lidar: Object containing lidar parameters.
        sweep_uuid:
    """

    annotations: Optional[Annotations]
    city_pose: Pose
    lidar: Lidar
    sweep_uuid: Tuple[str, int]
    is_ground: Optional[NDArrayByte] = None

    @classmethod
    def from_rust(cls, sweep: rust.Sweep) -> Sweep:
        annotations = Annotations(dataframe=sweep.annotations.to_pandas())
        city_pose = Pose(dataframe=sweep.city_pose.to_pandas())
        lidar = Lidar(dataframe=sweep.lidar.to_pandas())
        return cls(annotations=annotations, city_pose=city_pose, lidar=lidar, sweep_uuid=sweep.sweep_uuid)


@dataclass(frozen=True)
class Pose:
    """Stores the annotations and lidar for one sweep."""

    dataframe: pd.DataFrame

    @cached_property
    def Rt(self) -> Tuple[Tensor, Tensor]:
        quat_wxyz: NDArrayFloat = self.dataframe[QUAT_WXYZ_FIELDS].to_numpy()
        translation: NDArrayFloat = self.dataframe[TRANSLATION_FIELDS].to_numpy()

        rotation = quat_to_mat(quat_wxyz)
        return torch.as_tensor(rotation, dtype=torch.float32), torch.as_tensor(translation, dtype=torch.float32)

    def SE3(self) -> SE3:
        R, t = self.Rt
        return SE3(rotation=R[0].numpy(), translation=t[0].numpy())


@dataclass(frozen=True)
class Flow:
    """Models scene flow pseudolabels for a LiDARSweep.

       Scene Flow pseudolabels come from the relative motion of all tracked objects
       between two sweeps.


    Args:
        flow: (N,3) Motion vectors (x,y,z) in meters.
        valid: (N,1) 1 if the flow was succesfuly estimated for that point 0 otherwise
        classes: (N,1) the semantic object class of each point (0 is background)
        ego_motion: SE3 the motion of the vehicle between the two sweeps
    """

    flow: Optional[NDArrayFloat]
    valid: Optional[NDArrayByte]
    classes: Optional[NDArrayByte]
    ego_motion: SE3

    def __len__(self) -> int:
        """Return the number of LiDAR returns in the aggregated sweep."""
        return int(self.flow.shape[0]) if self.flow is not None else 0

    @classmethod
    def from_sweep_pair(cls, sweeps: Tuple[Sweep, Sweep]):
        poses = [sweep.city_pose.SE3() for sweep in sweeps]
        ego1_SE3_ego0 = poses[1].inverse().compose(poses[0])
        if sweeps[0].annotations is None or sweeps[1].annotations is None:
            return cls(flow=None, valid=None, classes=None, ego_motion=ego1_SE3_ego0)
        else:
            annotations: List[Annotations] = [sweeps[0].annotations, sweeps[1].annotations]
        
        cuboids = [annotations_to_id_cudboid_map(anno) for anno in annotations]
        pcs = [sweep.lidar.dataframe[['x', 'y', 'z']].to_numpy() for sweep in sweeps]
        
        
        # Convert to float32s
        ego1_SE3_ego0.rotation = ego1_SE3_ego0.rotation.astype(np.float32)
        ego1_SE3_ego0.translation = ego1_SE3_ego0.translation.astype(np.float32)
        
        flow = ego1_SE3_ego0.transform_point_cloud(pcs[0]) -  pcs[0]
        # Convert to float32s
        flow = flow.astype(np.float32)
        
        valid = np.ones(len(pcs[0]), dtype=np.uint8)
        classes = np.zeros(len(pcs[0]), dtype=np.uint8)
        
        
        for id in cuboids[0]:
            c0 = cuboids[0][id]
            c0.length_m += 0.2 # the bounding boxes are a little too tight and some points are missed
            c0.width_m += 0.2
            obj_pts, obj_mask = c0.compute_interior_points(pcs[0])
            classes[obj_mask] = CATEGORY_MAP[c0.category] + 1
        
            if id in cuboids[1]:
                c1 = cuboids[1][id]
                c1_SE3_c0 = c1.dst_SE3_object.compose(c0.dst_SE3_object.inverse())
                obj_flow = c1_SE3_c0.transform_point_cloud(obj_pts) - obj_pts
                flow[obj_mask] = obj_flow.astype(np.float32)
            else:
                valid[obj_mask] = 0
        return cls(flow=flow, valid=valid, classes=classes, ego_motion=ego1_SE3_ego0)
    

    
def prevent_fsspec_deadlock() -> None:
    """Reset the fsspec global lock to prevent deadlocking in forked processes."""
    fsspec.asyn.reset_lock()


def query_pose(poses: pd.DataFrame, timestamp_ns: int) -> SE3:
    """Query the SE(3) transformation as the provided timestamp in nanoseconds.

    Args:
        poses: DataFrame of quaternion and translation components.
        timestamp_ns: Timestamp of interest in nanoseconds.

    Returns:
        SE(3) at timestamp_ns.
    """
    mask = poses.loc[:, "timestamp_ns"] == timestamp_ns
    pose = poses.loc[mask, ["qw", "qx", "qy", "qz", "tx_m", "ty_m", "tz_m"]]
    pose_npy: NDArrayFloat = pose.to_numpy().squeeze()
    quat = pose_npy[:4]
    translation = pose_npy[4:]
    return SE3(
        rotation=quat_to_mat(quat),
        translation=translation,
    )


def compute_interior_points_mask(xyz_m: Tensor, cuboid_vertices: Tensor) -> Tensor:
    r"""Compute the interior points within a set of _axis-aligned_ cuboids.

    Reference:
        https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d
            5------4
            |\\    |\\
            | \\   | \\
            6--\\--7  \\
            \\  \\  \\ \\
        l    \\  1-------0    h
         e    \\ ||   \\ ||   e
          n    \\||    \\||   i
           g    \\2------3    g
            t      width.     h
             h.               t.

    Args:
        xyz_m: (N,3) Points in Cartesian space.
        cuboid_vertices: (K,8,3) Vertices of the cuboids.

    Returns:
        (N,) A tensor of boolean flags indicating whether the points
            are interior to the cuboid.
    """
    vertices = cuboid_vertices[:, [6, 3, 1]]
    uvw = cuboid_vertices[:, 2:3] - vertices
    reference_vertex = cuboid_vertices[:, 2:3]

    dot_uvw_reference = uvw @ reference_vertex.transpose(1, 2)
    dot_uvw_vertices = torch.diagonal(uvw @ vertices.transpose(1, 2), 0, 2)[..., None]
    dot_uvw_points = uvw @ xyz_m.T

    constraint_a = torch.logical_and(dot_uvw_reference <= dot_uvw_points, dot_uvw_points <= dot_uvw_vertices)
    constraint_b = torch.logical_and(dot_uvw_reference >= dot_uvw_points, dot_uvw_points >= dot_uvw_vertices)
    is_interior: Tensor = torch.logical_or(constraint_a, constraint_b).all(dim=1)
    return is_interior


def annotations_to_id_cudboid_map(annotations: Annotations):
    ids = annotations.dataframe.track_uuid.to_numpy()
    annotations_df_with_ts = annotations.dataframe.assign(timestamp_ns = None)
    cuboid_list = CuboidList.from_dataframe(annotations_df_with_ts)

    cuboids_and_ids = dict(zip(ids, cuboid_list.cuboids))
    return cuboids_and_ids

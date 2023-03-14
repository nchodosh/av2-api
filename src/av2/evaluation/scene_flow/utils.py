from av2.torch.dataloaders.scene_flow import SceneFlowDataloader
from av2.torch.dataloaders.utils import Sweep, Flow
from typing import Tuple


def get_eval_subset(dataloader: SceneFlowDataloader):
    return list(range(len(dataloader)))[::5]

def get_eval_point_mask(datum: Tuple[Sweep, Sweep, Flow]):
    pcl = datum[0].lidar.as_tensor()
    is_close =  (pcl[:, 0].abs() <= 50) & (pcl[:, 1].abs() <= 5)

    if datum[0].is_ground is None:
        raise ValueError('Must have ground annotations loaded to determine eval mask')

    return (is_close & ~datum[0].is_ground).numpy()
                                
                                

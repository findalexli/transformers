from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from .utils import TensorType, logging
from .image_utils import ChannelDimension

logger = logging.get_logger(__name__)

def get_tensor_size(image: torch.Tensor, channel_dim: Optional[ChannelDimension] = None) -> Tuple[int, int]:
    """Get size of tensor image (height, width)"""
    if channel_dim is None:
        if image.shape[0] == 3:  # Assume CHW format
            channel_dim = ChannelDimension.FIRST
        else:  # Assume HWC format
            channel_dim = ChannelDimension.LAST
            
    if channel_dim == ChannelDimension.FIRST:
        return image.shape[1:3]
    return image.shape[0:2]

def to_tensor_channel_dimension(
    image: torch.Tensor,
    channel_dim: ChannelDimension,
    input_channel_dim: Optional[ChannelDimension] = None
) -> torch.Tensor:
    """Convert tensor to specified channel dimension format"""
    if input_channel_dim is None:
        if image.shape[0] == 3:
            input_channel_dim = ChannelDimension.FIRST
        else:
            input_channel_dim = ChannelDimension.LAST
            
    if channel_dim == input_channel_dim:
        return image
        
    if channel_dim == ChannelDimension.FIRST:
        return image.permute(2, 0, 1)
    return image.permute(1, 2, 0)

class BatchTensorFeature:
    """
    Batch feature class for tensor data
    """
    def __init__(
        self,
        data: Dict[str, List[torch.Tensor]],
        tensor_type: Optional[Union[str, TensorType]] = None
    ):
        self.data = data
        self.tensor_type = tensor_type
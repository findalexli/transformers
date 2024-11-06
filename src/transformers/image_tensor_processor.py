from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
# from .image_pr import (
#     get_tensor_size,
#     to_tensor_channel_dimension,
#     BatchTensorFeature
# )
from .image_processing_base import BatchFeature, ImageProcessingMixin
from .image_tensor_utils import ChannelDimension
from .utils import logging

logger = logging.get_logger(__name__)

class BaseImageTensorProcessor(ImageProcessingMixin):
    """Base class for image processing using PyTorch tensors"""
    
    def __init__(self, **kwargs):
        self.config = kwargs

    # TODO: Correct resizing logic
    def resize_tensor(
        self,
        image: torch.Tensor,
        size: Tuple[int, int],
        interpolation: str = "bilinear",
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[ChannelDimension] = None,
    ) -> torch.Tensor:
        """Resize tensor image"""
        if input_data_format is None:
            if image.shape[0] == 3:
                input_data_format = ChannelDimension.FIRST
            else:
                input_data_format = ChannelDimension.LAST

        # Convert to channels first if needed
        if input_data_format == ChannelDimension.LAST:
            image = image.permute(2, 0, 1)

        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Resize
        image = F.interpolate(image, size=size, mode=interpolation, align_corners=False)
        
        # Remove batch dimension
        image = image.squeeze(0)

        # Convert back to original format if needed
        if input_data_format == ChannelDimension.LAST:
            image = image.permute(1, 2, 0)

        return image

    def normalize_tensor(
        self,
        image: torch.Tensor,
        mean: Union[float, List[float]],
        std: Union[float, List[float]],
        data_format: Optional[ChannelDimension] = None,
        input_data_format: Optional[ChannelDimension] = None,
    ) -> torch.Tensor:
        """Normalize tensor image"""
        mean = torch.tensor(mean, device=image.device)
        std = torch.tensor(std, device=image.device)

        if input_data_format is None:
            if image.shape[0] == 3:
                input_data_format = ChannelDimension.FIRST
            else:
                input_data_format = ChannelDimension.LAST

        if input_data_format == ChannelDimension.FIRST:
            mean = mean.view(-1, 1, 1)
            std = std.view(-1, 1, 1)
        else:
            mean = mean.view(1, 1, -1)
            std = std.view(1, 1, -1)

        return (image - mean) / std

    def rescale_tensor(
        self,
        image: torch.Tensor,
        scale: float,
        input_data_format: Optional[ChannelDimension] = None,
    ) -> torch.Tensor:
        """Rescale tensor image"""
        return image * scale

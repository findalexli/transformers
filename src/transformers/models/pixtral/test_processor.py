# import unittest
# from PIL import Image
# import numpy as np
# from transformers.models.pixtral.image_processing_pixtral import PixtralImageProcessor
#
# class TestPixtralImageProcessor(unittest.TestCase):
#     def setUp(self):
#         # Create a sample image
#         self.image = Image.new("RGB", (512, 512), color="white")
#         self.processor = PixtralImageProcessor()
#
#     def test_preprocess(self):
#         # Preprocess the sample image
#         processed_image = self.processor.preprocess(images=self.image)
#         self.assertIsNotNone(processed_image)
#
# if __name__ == "__main__":
#     unittest.main()


import numpy as np
from PIL import Image
import torch

from transformers.models.pixtral.image_processing_pixtral import PixtralImageProcessor
# from transformers.models.pixtral.image_processing_pixtral import PixtralImageTensorProcessor
# Sample Image Creation
# Create a simple RGB image as a NumPy array with pixel values between 0 and 255


image_inp = np.load('image_array.npy')
image_tensor = torch.from_numpy(image_inp).to(torch.float64)
image_tensor = image_tensor.unsqueeze(0)
# Set PyTorch print options to show more decimal places
torch.set_printoptions(precision=8)
#
# print("NumPy array:\n", image_inp)
# print("Tensor:\n", image_tensor)

# Instantiate the processor with default or custom parameters
processor = PixtralImageProcessor(
    do_resize=True,
    do_rescale=False,
    size={"longest_edge": 1024},  # Resize longest edge to 256 pixels
    do_normalize=True,
    input_data_format = "channels_last",
    data_format = "channels_first",
)

# Preprocess the image with specified settings
print("INPUT IMAGE TENSOR",image_tensor.shape)
preprocessed_output = processor.preprocess(images=image_tensor)

# Inspect the output
print("Preprocessed output:")
# print(preprocessed_output)
print("Pixel values shape:", preprocessed_output['pixel_values'][0][0].shape)
print("Image sizes:", preprocessed_output['image_sizes'])

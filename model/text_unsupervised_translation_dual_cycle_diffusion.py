import torch
import torch.nn as nn
import torchvision.transforms as transforms

from .model_utils import requires_grad
from .gan_wrapper.get_gan_wrapper import get_gan_wrapper


class TextUnsupervisedTranslation(nn.Module):

    def __init__(self, args):
        super(TextUnsupervisedTranslation, self).__init__()

        # Set up gan_wrapper
        self.gan_wrapper = get_gan_wrapper(args.gan)
        # Freeze.
        requires_grad(self.gan_wrapper, True)  # Otherwise, no trainable params.

        self.encode_transform = transforms.Compose([
            transforms.Resize(self.gan_wrapper.resolution),
            transforms.ToTensor()
        ])

    def forward(self, sample_id, original_image, encode_text, decode_text):
        # Eval mode for gan_wrapper.
        self.gan_wrapper.eval()

        assert not self.training

        img, _, _ = self.gan_wrapper(image=original_image, 
                               encode_text=encode_text,
                               decode_text=decode_text,
                               return_mask=True, 
                               return_img=True)

        # Placeholders
        losses = dict()
        weighted_loss = torch.zeros_like(sample_id).float()

        
        if type(mid_img)==list:
            mid_img = torch.stack(mid_img, dim=1)

        return (original_image, img), weighted_loss, losses

    @property
    def device(self):
        return next(self.parameters()).device


Model = TextUnsupervisedTranslation

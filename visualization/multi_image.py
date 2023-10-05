import os
import math
import torch
import torch.nn.functional as F

from utils.file_utils import save_images


class Visualizer(object):

    def __init__(self, args):
        self.args = args

    def visualize(self,
                  images,
                  model,
                  description: str,
                  save_dir: str,
                  step: int,
                  ):
        # Merge.
        k = len(images)
        assert k >= 2
        bsz, c, h, w = images[0].shape
        if k == 3:
            bsz2, c2, h2, w2 = images[2].shape
            assert bsz == bsz2 and c == c2
            assert h2 == w2
            assert h == w
            if h == h2:
                pass
            else:
                assert h2 < h
                images = (
                    images[0],
                    images[1],
                    F.interpolate(images[2], size=(h, w), mode='nearest'),
                )
        if k > 3:
            mask = images[2]
            mid_img = images[3]
            print('mask and mid_img shape', mask.shape, mid_img.shape, k)
            images = [images[0], images[1]]
            k = 2 + mask.shape[1] + mid_img.shape[1]
            for m_i in range(mid_img.shape[1]):
                images.append(mid_img[:, m_i])
            for m_i in range(mask.shape[1]):
                tmp = F.interpolate(mask[:, m_i], size=(h, w), mode='nearest')
                tmp = tmp.expand(-1,3,-1,-1)
                images.append(tmp)
            
        images = torch.stack(images, dim=1).view(bsz * k, c, h, w)

        # Just visualize the first 64 images.
        if k > 3:
            images = images[:20 * k, :, :, :]
        else:
            images = images[:100 * k, :, :, :]

        if k > 3:
            save_images(
                images,
                output_dir=save_dir,
                file_prefix=description,
                nrows=7,
                iteration=step,
            )
        else:
            save_images(
                images,
                output_dir=save_dir,
                file_prefix=description,
                nrows=8,
                iteration=step,
            )

        # Lower resolution
        # images_256 = F.interpolate(
        #     images,
        #     (256, 256),
        #     mode='bicubic',
        # )
        # save_images(
        #     images_256,
        #     output_dir=save_dir,
        #     file_prefix=f'{description}_256',
        #     nrows=8,
        #     iteration=step,
        # )



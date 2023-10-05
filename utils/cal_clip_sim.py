from model.clip_extractor import ClipExtractor

from PIL import Image
from torchvision.utils import save_image
import torch
import torchvision.transforms as T
from einops import rearrange
import torch.nn.functional as F


def load_image(path, size=224):
    if type(path)==str:
        img = Image.open(path).convert('RGB')
    else:   # type(path)==Image:
        img = path.convert('RGB')
    tsf = T.Compose(
            [
                # we added interpolation to CLIP positional embedding, allowing to work with arbitrary resolution.
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]),
            ]
        )
    return tsf(img).unsqueeze(0)

def process_image(img, size=224):
    img = F.interpolate(img, size=(size, size), mode='bilinear')
    img = torch.clamp(img,min=0,max=1)
    mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1,3,1,1).to(img.device) 
    std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1,3,1,1).to(img.device)
    img = (img - mean) / std
    return img

def get_img_feature_map(img, clip_extractor, size):
    n = size // 224
    assert size%224==0, "size need be divided"
    out = []
    for i in range(n):
        for j in range(n):
            tmp = img[:,:,224*i:224*(i+1),224*j:224*(j+1)]
            tmp = clip_extractor.get_img_feature_map_with_tensor(tmp)[:, 1:]
            tmp = rearrange(tmp, 'b (h w) c -> b c h w', h=7, w=7)
            out.append(tmp.unsqueeze(2))
    out = torch.cat(out, dim=2)
    out = rearrange(out, 'b c (n1 n2) h w -> b c (n1 h) (n2 w)', n1=n, n2=n)
    return out
    
def cal_diff(f1, f2, mode=0, mask_size=64):
    f1n = f1.norm(dim=1, keepdim=True)
    f2n = f2.norm(dim=1, keepdim=True)

    if mode==0:
        diff = f1n - f2n
    else:
        diff = (f1 * f2).sum(dim=1, keepdim=True)/torch.clamp((f1n * f2n).sum(dim=1, keepdim=True), min=1e-08)
    diff_abs = diff.abs()
    diff_abs = (diff_abs - diff_abs.min())/(diff_abs - diff_abs.min()).max()
    diff_abs = diff_abs.mean(dim=1, keepdim=True)
    diff_abs = torch.nn.functional.interpolate(diff_abs, mask_size)
    if mode==1:
        diff_abs = 1 - diff_abs
    return diff_abs


class MaskGenerator():
    def __init__(self):
        self.clip_extractor = ClipExtractor()

    def cal_mask(self, img1, img2, img_size=448, mask_size=64):
        if type(img1)!=list:
            imgs1 = [img1]
        else:
            imgs1 = img1
        if type(img2)!=list:
            imgs2 = [img2]
        else:
            imgs2 = img2
        
        out = []
        for i in range(len(imgs1)):
            img1, img2 = imgs1[i], imgs2[i]
            img1 = load_image(img1, img_size) 
            sim1 = get_img_feature_map(img1, self.clip_extractor, img_size)  
            img2 = load_image(img2, img_size) 
            sim2 = get_img_feature_map(img2, self.clip_extractor, img_size)  
            diff_abs = cal_diff(sim1, sim2, mode=1, mask_size=mask_size)
            out.append(diff_abs)
        out = torch.cat(out, dim=0)
        
        return out

    def cal_mask_with_tensor(self, img1, img2, img_size=448, mask_size=64):
        img1 = process_image(img1, size=img_size)
        img2 = process_image(img2, size=img_size)
        sim1 = get_img_feature_map(img1, self.clip_extractor, img_size)
        sim2 = get_img_feature_map(img2, self.clip_extractor, img_size)
        out = []
        for i in range(sim1.shape[0]):
            diff_abs = cal_diff(sim1[i].unsqueeze(0), sim2[i].unsqueeze(0), mode=1, mask_size=mask_size)
            out.append(diff_abs)
        out = torch.cat(out, dim=0)
        return out
        
       

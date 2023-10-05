import os
import argparse
import sys
sys.path.append(os.path.abspath('model/lib/latentdiff'))
import glob
from omegaconf import OmegaConf
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

from model.energy.clean_clip import DirectionalCLIP
from txt2img import load_model_from_config 
from ldm.models.diffusion.ddim_dual_cycle import DDIMSampler
from ..model_utils import requires_grad
from utils.cal_clip_sim import MaskGenerator
import PIL
import cv2

def prepare_latentdiff_text(source_model_type):
    print('First of all, when the code changes, make sure that no part in the model is under no_grad!')

    if source_model_type == "text2img-large":
        config = OmegaConf.load(os.path.join('ckpts', 'ldm_models', source_model_type, 'txt2img-1p4B-eval.yaml'))

        latentdiff_ckpt = os.path.join('ckpts', 'ldm_models', source_model_type, 'model.ckpt')
    else:
        raise ValueError()

    return config, latentdiff_ckpt


def get_condition(model, text, bs):
    assert isinstance(text, list)
    assert isinstance(text[0], str)
    uc = model.get_learned_conditioning(bs * [""])
    print("model.cond_stage_key: ", model.cond_stage_key)
    c = model.get_learned_conditioning(text)
    # print("c.shape: ", c.shape)
    print('-' * 50)
    return c, uc


def convsample_ddim_conditional(model, steps, shape, x_T, skip_steps, eta, eps_list, scale, text):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    c, uc = get_condition(model, text, bs)
    samples, intermediates = ddim.sample_with_eps(steps,
                                                  eps_list,
                                                  conditioning=c,
                                                  batch_size=bs,
                                                  shape=shape,
                                                  eta=eta,
                                                  verbose=False,
                                                  x_T=x_T,
                                                  skip_steps=skip_steps,
                                                  unconditional_guidance_scale=scale,
                                                  unconditional_conditioning=uc,
                                                  log_every_t=1
                                                  )
    return samples, intermediates


def make_convolutional_sample_with_eps_conditional(model, custom_steps, eta, x_T, skip_steps, eps_list,
                                                   scale, text):
    with model.ema_scope("Plotting"):
        sample, intermediates = convsample_ddim_conditional(model,
                                                            steps=custom_steps,
                                                            shape=x_T.shape,
                                                            x_T=x_T,
                                                            skip_steps=skip_steps,
                                                            eta=eta,
                                                            eps_list=eps_list,
                                                            scale=scale,
                                                            text=text)

    x_sample = model.decode_first_stage(sample)

    return x_sample, intermediates


def ddpm_ddim_encoding_conditional(model, steps, shape, eta, white_box_steps, skip_steps, x0, scale, text):
    with model.ema_scope("Plotting"):
        ddim = DDIMSampler(model)
        bs = shape[0]
        shape = shape[1:]
        c, uc = get_condition(model, text, bs)

        z_list, x_list = ddim.ddpm_ddim_encoding(steps,
                                         conditioning=c,
                                         batch_size=bs,
                                         shape=shape,
                                         eta=eta,
                                         white_box_steps=white_box_steps,
                                         skip_steps=skip_steps,
                                         verbose=True,
                                         x0=x0,
                                         unconditional_guidance_scale=scale,
                                         unconditional_conditioning=uc,
                                         )

    return z_list, x_list


class LatentDiffStochasticTextWrapper(torch.nn.Module):

    def __init__(self, source_model_type, custom_steps, eta, white_box_steps, skip_steps,
                 encoder_unconditional_guidance_scales=None, decoder_unconditional_guidance_scales=None,
                 n_trials=None, ckpt=None, rank_mode='dclip'):
        super(LatentDiffStochasticTextWrapper, self).__init__()

        self.encoder_unconditional_guidance_scales = encoder_unconditional_guidance_scales
        self.decoder_unconditional_guidance_scales = decoder_unconditional_guidance_scales
        self.n_trials = n_trials

        # Set up generator
        self.config, self.ckpt = prepare_latentdiff_text(source_model_type)
        print(self.config)

        self.generator = load_model_from_config(self.config, self.ckpt, verbose=True)

        print(75 * "-")

        self.eta = eta
        self.custom_steps = custom_steps
        self.white_box_steps = white_box_steps
        self.skip_steps = skip_steps
        self.rank_mode = rank_mode

        self.resolution = self.generator.first_stage_model.encoder.resolution
        print(f"resolution: {self.resolution}")

        print(f'Using DDIM sampling with {self.custom_steps} sampling steps and eta={self.eta}')

        # Freeze.
        # requires_grad(self.generator, False)

        # Post process.
        self.post_process = transforms.Compose(  # To un-normalize from [-1.0, 1.0] (GAN output) to [0, 1]
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
        )

        # Directional CLIP score.
        self.directional_clip = DirectionalCLIP()

    def generate(self, z_ensemble, decode_text):
        img_ensemble = []
        intermediates_ensemble = []

        for i, z in enumerate(z_ensemble):
            skip_steps = self.skip_steps[i % len(self.skip_steps)]
            bsz = z.shape[0]
            if self.white_box_steps != -1:
                eps_list = z.view(bsz, (self.white_box_steps - skip_steps), self.generator.channels, self.generator.image_size, self.generator.image_size)
            else:
                eps_list = z.view(bsz, 1, self.generator.channels, self.generator.image_size, self.generator.image_size)
            x_T = eps_list[:, 0]
            eps_list = eps_list[:, 1:]

            for decoder_unconditional_guidance_scale in self.decoder_unconditional_guidance_scales:
                img, intermediates = make_convolutional_sample_with_eps_conditional(self.generator,
                                                                     custom_steps=self.custom_steps,
                                                                     eta=self.eta,
                                                                     x_T=x_T,
                                                                     skip_steps=skip_steps,
                                                                     eps_list=eps_list,
                                                                     scale=decoder_unconditional_guidance_scale,
                                                                     text=decode_text)
                img_ensemble.append(img)
                intermediates = torch.stack(intermediates['x_inter'], dim=1).view(bsz, -1)
                intermediates_ensemble.append(intermediates)

        return img_ensemble, intermediates_ensemble

    def encode(self, image, encode_text):
        # Eval mode for the generator.
        self.generator.eval()

        # Normalize.
        image = (image - 0.5) * 2.0
        # Resize.
        assert image.shape[2] == image.shape[3] == self.resolution
        with torch.no_grad():
            # Encode.
            encoder_posterior = self.generator.encode_first_stage(image)
            z = self.generator.get_first_stage_encoding(encoder_posterior)
            x0 = z

        bsz = image.shape[0]
        z_ensemble = []
        x_ensemble = []
        for trial in range(self.n_trials):
            for encoder_unconditional_guidance_scale in self.encoder_unconditional_guidance_scales:
                for skip_steps in self.skip_steps:
                    with torch.no_grad():
                        # DDIM forward.
                        z_list, x_list = ddpm_ddim_encoding_conditional(self.generator,
                                                                steps=self.custom_steps,
                                                                shape=x0.shape,
                                                                eta=self.eta,
                                                                white_box_steps=self.white_box_steps,
                                                                skip_steps=skip_steps,
                                                                x0=x0,
                                                                scale=encoder_unconditional_guidance_scale,
                                                                text=encode_text)
                        z = torch.stack(z_list, dim=1).view(bsz, -1)
                        z_ensemble.append(z)
                        x_tmp = torch.stack(x_list, dim=1).view(bsz, -1)
                        x_ensemble.append(x_tmp)

        return z_ensemble, x_ensemble

    def forward(self, z_ensemble, original_img, encode_text, decode_text, return_score=False):
        # Eval mode for the generator.
        self.generator.eval()

        img_ensemble, intermediates_ensemble = self.generate(z_ensemble, decode_text)
        assert len(img_ensemble) == len(self.decoder_unconditional_guidance_scales) * len(self.encoder_unconditional_guidance_scales) * len(self.skip_steps) * self.n_trials

        # Post process.
        img_ensemble = [self.post_process(img) for img in img_ensemble]

        # Rank with directional CLIP score.
        score_ensemble = []
        for img in img_ensemble:
            clip_score, dclip_score = self.directional_clip(img, original_img, encode_text, decode_text)
            if self.rank_mode=="clip":
                dclip_score = clip_score
            assert dclip_score.shape == (img.shape[0],)

            score_ensemble.append(dclip_score)
        score_ensemble = torch.stack(score_ensemble, dim=1)  # (bsz, n_trials)
        assert score_ensemble.shape == (img_ensemble[0].shape[0], len(img_ensemble))

        best_idx = torch.argmax(score_ensemble, dim=1)  # (bsz,)
        bsz = score_ensemble.shape[0]
        img = torch.stack(
            [
                img_ensemble[best_idx[b].item()][b] for b in range(bsz)
            ],
            dim=0,
        )
        score_out = torch.stack(
            [
                score_ensemble[b][best_idx[b].item()] for b in range(bsz)
            ],
            dim=0,
        )
        intermediates = [
                intermediates_ensemble[best_idx[b].item()][b].unsqueeze(0) for b in range(bsz)
            ]
        print('best scales:')
        best_idx = best_idx % (len(self.decoder_unconditional_guidance_scales) * len(self.encoder_unconditional_guidance_scales) * len(self.skip_steps))
        print(
            [
                (
                    self.encoder_unconditional_guidance_scales[_best_idx // (len(self.decoder_unconditional_guidance_scales) * len(self.skip_steps))],
                    self.decoder_unconditional_guidance_scales[_best_idx % (len(self.decoder_unconditional_guidance_scales) * len(self.skip_steps)) // len(self.skip_steps)],
                    self.skip_steps[_best_idx % len(self.skip_steps)],
                )
                for _best_idx in best_idx
            ]
        )
        scales = [
                (
                    self.encoder_unconditional_guidance_scales[_best_idx // (len(self.decoder_unconditional_guidance_scales) * len(self.skip_steps))],
                    self.decoder_unconditional_guidance_scales[_best_idx % (len(self.decoder_unconditional_guidance_scales) * len(self.skip_steps)) // len(self.skip_steps)],
                    self.skip_steps[_best_idx % len(self.skip_steps)],
                )
                for _best_idx in best_idx
            ]
        
        if return_score:
            return img, intermediates, scales, score_out
        return img, intermediates, scales

    @property
    def device(self):
        return next(self.parameters()).device


def cal_diff(ref_img, query_img):
    ## Getting difference between the denoised images
    # Taking the difference
    diff_img = ref_img-query_img
    # diff_img.shape

    # Taking mean of all channels
    diff_mean = diff_img.mean(dim=1)
    # diff_mean.shape

    # normalize (between 0 to 1)
    diff_normed = (diff_mean - diff_mean.min())/(diff_mean - diff_mean.min()).max()
    # diff_normed

    # binarize (0 or 1)
    diff_bin=(diff_normed>0.5).float()
    return diff_normed, diff_bin

def load_img(path):
    if type(path)==str:
        image = PIL.Image.open(path).convert("RGB")
    else:
        image = path
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    # image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = image.resize((256, 256), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image # 2.*image - 1.

def extract_channel_mask(img, do_inverse=False):
    kernel = np.ones((3,3),np.uint8)
    img = (img*255).squeeze().cpu().to(torch.uint8).numpy()
    if do_inverse:
        ret2,img2 = cv2.threshold(img,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    else:
        ret2,img2 = cv2.threshold(img,0,1,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    dilate = cv2.dilate(img2, kernel)
    return dilate


class DualCycleDiffusion(torch.nn.Module):
    def __init__(self, 
                 source_model_type, custom_steps, eta, white_box_steps,
                 encoder_unconditional_guidance_scales=None, decoder_unconditional_guidance_scales=None,
                 n_trials=1, ckpt=None, skip_steps=None,
                 mask_mode=0, n_trials_dual=1,
                 rank_mode='dclip', mask_idx=0, mask_skip_step_trunc=-1,
                 force_mask=False):
        super(DualCycleDiffusion, self).__init__()
        
        self.sds = LatentDiffStochasticTextWrapper(source_model_type, custom_steps, eta, white_box_steps, skip_steps,
                 encoder_unconditional_guidance_scales=encoder_unconditional_guidance_scales, decoder_unconditional_guidance_scales=decoder_unconditional_guidance_scales,
                 n_trials=n_trials, ckpt=ckpt, rank_mode=rank_mode)
        self.mask_generator = MaskGenerator()
        self.white_box_steps = white_box_steps
        self.sds.eval()
        self.resolution = 256
        print(f"resolution: {self.resolution}")
        self.n_trials_dual = n_trials_dual
        self.mask_mode = mask_mode
        self.mask_idx = mask_idx
        self.mask_skip_step_trunc = mask_skip_step_trunc
        self.force_mask = force_mask

    
    def set_skip_step(self, skip_steps=[15]):
        self.sds.skip_steps = skip_steps  # self.ref_step * len(self.sds.skip_steps)
    
    def process(self, init_img, encode_text, decode_text,
                tag=''):
        sds = self.sds
        mask_generator = self.mask_generator
        device = self.device
        skip_steps = sds.skip_steps
        white_box_steps = self.white_box_steps

        if type(encode_text)==str:
            encode_text = [encode_text] # * batch_size
        if type(decode_text)==str:
            decode_text = [decode_text]


        init_image = init_img # load_img(init_img).to(device)
        # init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        bsz = init_image.shape[0]
        
        trials = self.n_trials_dual

        with torch.no_grad():
            print('start encode')
            
            sds_encoder_scales_ori = sds.encoder_unconditional_guidance_scales
            sds_decoder_scales_ori = sds.decoder_unconditional_guidance_scales
            sds_skip_steps_ori = sds.skip_steps
            sds_skip_steps_ori.sort()
            sds_trials_ori = sds.n_trials
            sds.n_trials = 1
            N = len(sds_encoder_scales_ori) * len(sds_decoder_scales_ori) * len(sds_skip_steps_ori)
            mask_all = [0 for _ in range(N)]
            img_all = [None] * bsz
            img_score = [-10000.0] * bsz
            scores_all = [[-10000.0]*bsz for _ in range(N)]
            z_ensemble_all = [[None]*bsz for _ in range(N)]
            x_ensemble_all = [[None]*bsz for _ in range(N)]
            intermediates_all = [[None]*bsz for _ in range(N)]
           
            for t_i in range(trials):
                for e_i in range(len(sds_encoder_scales_ori)):
                    sds.encoder_unconditional_guidance_scales = [sds_encoder_scales_ori[e_i]]
                    for s_i in range(len(sds_skip_steps_ori)):
                        sds.skip_steps = [sds_skip_steps_ori[s_i]]
                        for d_i in range(len(sds_decoder_scales_ori)):
                            sds.decoder_unconditional_guidance_scales = [sds_decoder_scales_ori[d_i]]
                            z_ensemble, x_ensemble = sds.encode(init_image, encode_text)
                            img, intermediates, _, score = sds(z_ensemble, init_image, encode_text, decode_text, return_score=True)
                            # print('score', score)
                            idx = e_i*len(sds_skip_steps_ori)*len(sds_decoder_scales_ori)+s_i*len(sds_decoder_scales_ori)+d_i
                            for b_i in range(bsz):
                                if scores_all[idx][b_i] < score[b_i]:
                                    z_ensemble_all[idx][b_i] = z_ensemble[0][b_i]
                                    x_ensemble_all[idx][b_i] = x_ensemble[0][b_i]
                                    intermediates_all[idx][b_i] = intermediates[b_i][0]
                                    scores_all[idx][b_i] = score[b_i]
                                if img_score[b_i] < score[b_i]:
                                    img_score[b_i] = score[b_i]
                                    img_all[b_i] = img[b_i]
                            # Caculate the mask with the CLIP encoder
                            mask = mask_generator.cal_mask_with_tensor(init_image, img, mask_size=32).to(self.device)
                            mask_all[idx] = mask_all[idx] + mask
                            


            mask_all = [mask/trials for mask in mask_all]
            mask_all = torch.stack(mask_all, dim=1)
            
            # reset
            sds.encoder_unconditional_guidance_scales = sds_encoder_scales_ori
            sds.decoder_unconditional_guidance_scales = sds_decoder_scales_ori
            sds.skip_steps = sds_skip_steps_ori
            sds.n_trials = sds_trials_ori
            if self.mask_skip_step_trunc > 0:
                mask_all = mask_all.view(bsz, len(sds_encoder_scales_ori), len(sds_skip_steps_ori), len(sds_decoder_scales_ori), 1, 64,64)
                mask_all = mask_all[:, :, :self.mask_skip_step_trunc]
                mask_all = mask_all.reshape(bsz, -1 , 1, 64, 64)
            mask_all = mask_all.mean(dim=1)
            img_all = torch.stack(img_all, dim=0)
        return mask_all, img_all, z_ensemble_all, x_ensemble_all, intermediates_all
    

    @property
    def device(self):
        return next(self.parameters()).device

    def getMask(self, init_img, encode_text, decode_text, return_img=False):
        # The forward path of BE-cycle
        mask, img, z_ensemble_all, x_ensemble_all, intermediates_all = self.process(init_img, encode_text, decode_text, tag="first_")
        # The inverted path of BE-cycle
        mask1, img1, _, _, _ = self.process(img, decode_text, encode_text, tag="second_")
        if self.mask_mode==0:
            mask_out = ((mask+mask1)>1).float()
        else:
            mask_out = (mask+mask1)/2.0
            tmp = []
            for t in range(mask_out.shape[0]):
                tmp.append(torch.from_numpy(extract_channel_mask(mask_out[t])).to(self.device).float())
            mask_out = torch.stack(tmp, dim=0)
        if return_img:
            return [mask_out, mask, mask1], [img, img1], [z_ensemble_all, x_ensemble_all, intermediates_all]
        return [mask_out, mask, mask1], [z_ensemble_all, x_ensemble_all, intermediates_all]
    
    def get_img_with_mask(self, init_img, encode_text, decode_text, mask, codes):
        sds = self.sds
        sds_encoder_scales_ori = sds.encoder_unconditional_guidance_scales
        sds_decoder_scales_ori = sds.decoder_unconditional_guidance_scales
        sds_skip_steps_ori = sds.skip_steps
        sds_skip_steps_ori.sort()
        sds.n_trials = 1
        white_box_steps = self.white_box_steps
        bzs = init_img.shape[0]
        scores = [-1000] * bzs
        imgs = [None] * bzs
        z_ensemble_all, x_ensemble_all, intermediates_all = codes
        for t_i in range(self.n_trials_dual):
            for e_i in range(len(sds_encoder_scales_ori)):
                en_scale = sds_encoder_scales_ori[e_i]
                sds.encoder_unconditional_guidance_scales = [en_scale]
                for s_i in range(len(sds_skip_steps_ori)):
                    skip_steps = sds_skip_steps_ori[s_i]
                    sds.skip_steps = [skip_steps]
                    for d_i in range(len(sds_decoder_scales_ori)):
                        de_scale = sds_decoder_scales_ori[d_i]
                        sds.decoder_unconditional_guidance_scales = [de_scale]
                        idx = e_i*len(sds_skip_steps_ori)*len(sds_decoder_scales_ori)+s_i*len(sds_decoder_scales_ori)+d_i
                        z_ensemble, x_ensemble = z_ensemble_all[idx], x_ensemble_all[idx]    #sds.encode(init_img, encode_text)
                        intermediates = intermediates_all[idx]
                        z_ensemble = [torch.stack(z_ensemble, dim=0)]
                        x_ensemble = [torch.stack(x_ensemble, dim=0)]
                        intermediates = [torch.stack(intermediates, dim=0)]
                        
                        if self.force_mask and s_i < len(sds_skip_steps_ori)-1:
                            s_i2_start = s_i + 1
                        else:
                            s_i2_start = s_i
                        for s_i2 in range(s_i2_start, len(sds_skip_steps_ori)):
                            query_step = sds_skip_steps_ori[s_i2]
                            z_ensemble_tmp = []
                            for i in range(len(z_ensemble)):
                                b = z_ensemble[i].shape[0]
                                z_ensemble[i] = z_ensemble[i].view(b, white_box_steps-skip_steps, 4, -1)
                                x_ensemble[i] = x_ensemble[i].view(b, white_box_steps-skip_steps, 4, -1)
                                intermediates[i] = intermediates[i].view(b, white_box_steps-skip_steps, 4, -1)
                                z_ensemble[i][:, query_step-skip_steps] = intermediates[i][:, query_step-skip_steps]*mask.view(bzs, 1, -1) + x_ensemble[i][:, query_step-skip_steps]*(1-mask.view(bzs, 1, -1))
                                z_ensemble_tmp.append(z_ensemble[i][:, query_step-skip_steps:].view(b, -1))
                            sds.skip_steps = [query_step]
                            img, _, _, score = sds(z_ensemble_tmp, init_img, encode_text, decode_text, return_score=True)
                            sds.skip_steps = [skip_steps]
                            for j in range(bzs):
                                if score[j] > scores[j]:    
                                    scores[j] = score[j]
                                    imgs[j] = img[j]
        imgs = torch.stack(imgs, dim=0)

        # reset
        sds.encoder_unconditional_guidance_scales = sds_encoder_scales_ori
        sds.decoder_unconditional_guidance_scales = sds_decoder_scales_ori
        sds.skip_steps = sds_skip_steps_ori
        return imgs    


    def forward(self, image, encode_text, decode_text, return_mask=False, return_img=False):
        with torch.no_grad():
            init_img = image
            # Calculate the unbiased mask
            if return_img:
                mask_list, mid_img_list, codes = self.getMask(init_img, encode_text, decode_text, return_img=return_img)
            else:
                mask_list, codes = self.getMask(init_img, encode_text, decode_text)
            mask = mask_list[self.mask_idx]
            if self.mask_idx > 0:
                mask = (mask>0.5).float()
            # Mask-guided unbiased editing
            img = self.get_img_with_mask(init_img, encode_text, decode_text, mask, codes)
            
        if return_mask:
            if return_img:
                return img, mask_list, mid_img_list
            else:
                return img, mask_list

        return img





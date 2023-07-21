import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
from pytorch_grad_cam import EigenCAM, GradCAM, AblationCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from mmseg.models.builder import build_segmentor

from mcode import select_device, UnNormalize

# config
ckpt_path = "/mnt/sdd/nguyen.van.quan/Researchs/Polyp/runs/test/checkpoints/model_50.pth"
image_path = "/mnt/sdd/nguyen.van.quan/Researchs/Polyp/TestDataset/CVC-300/images/150.png"
mask_path = "/mnt/sdd/nguyen.van.quan/Researchs/Polyp/TestDataset/CVC-300/masks/150.png"
transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
model_cfg = dict(
    type='SunSegmentor',
    backbone=dict(
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        pretrained=None),
    decode_head=dict(
        type='UPerHeadV3',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0))
)
target_layers = [
    "model.decode_head.fusion_conv"
]

class SemanticSegmentationTarget:
    """ Gets a binary spatial mask and a category,
        And return the sum of the category scores,
        of the pixels in the mask. """

    def __init__(self, mask):
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[:, :, :] * self.mask).sum()


if __name__ == '__main__':
    # init
    device = select_device('')

    # model
    model = build_segmentor(model_cfg)
    model.load_state_dict(torch.load(ckpt_path, map_location='cuda:0'))
    model.to(device)
    model.eval()

    target_layers = [eval(target_layer) for target_layer in target_layers]

    image = cv2.imread(image_path)
    image = cv2.resize(image, (352, 352))[:, :, ::-1]
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (352, 352))[:, :, 0]
    sample = transform(image=image, mask=mask)
    img, gt_mask = sample["image"], sample["mask"]
    gt_mask = np.asarray(gt_mask, np.float32)
    img = img[None].to(device)

    with torch.no_grad():
        res = model(img)[0]
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        pred = res.round()

    targets = [SemanticSegmentationTarget(pred)]
    with AblationCAM(model=model,
                     target_layers=target_layers,
                     use_cuda=torch.cuda.is_available(),
                     batch_size=1) as cam:
        grayscale_cam = cam(input_tensor=img,
                            targets=targets)[0, :]
        cam_image = show_cam_on_image(image / 255, grayscale_cam, use_rgb=True, image_weight=0.8)
        plt.figure(figsize=(30, 10))
        plt.subplot(1, 3, 1)
        plt.imshow(cam_image)
        plt.subplot(1, 3, 2)
        plt.imshow(np.repeat(np.expand_dims(pred, axis=-1), repeats=3, axis=-1))
        plt.subplot(1, 3, 3)
        plt.imshow(np.repeat(np.expand_dims(gt_mask, axis=-1), repeats=3, axis=-1))
        plt.show()
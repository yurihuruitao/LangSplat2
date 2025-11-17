import os
import random
import argparse

import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2

from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy

import torch
import torchvision
from torch import nn

try:
    import open_clip
except ImportError:
    assert False, "open_clip is not installed, install it with `pip install open-clip-torch`"


@dataclass
class OpenCLIPNetworkConfig:
    _target: Type = field(default_factory=lambda: OpenCLIPNetwork)
    clip_model_type: str = "ViT-B-16"
    clip_model_pretrained: str = "laion2b_s34b_b88k"
    clip_n_dims: int = 512
    negatives: Tuple[str] = ("object", "things", "stuff", "texture")
    positives: Tuple[str] = ("",)

class OpenCLIPNetwork(nn.Module):
    def __init__(self, config: OpenCLIPNetworkConfig):
        super().__init__()
        self.config = config
        self.process = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        model, _, _ = open_clip.create_model_and_transforms(
            self.config.clip_model_type,  # e.g., ViT-B-16
            pretrained=self.config.clip_model_pretrained,  # e.g., laion2b_s34b_b88k
            precision="fp16",
        )
        model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.config.clip_model_type)
        self.model = model.to("cuda")
        self.clip_n_dims = self.config.clip_n_dims

        self.positives = self.config.positives    
        self.negatives = self.config.negatives
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = model.encode_text(tok_phrases)
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.negatives]).to("cuda")
            self.neg_embeds = model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)
        self.neg_embeds /= self.neg_embeds.norm(dim=-1, keepdim=True)

        assert (
            self.pos_embeds.shape[1] == self.neg_embeds.shape[1]
        ), "Positive and negative embeddings must have the same dimensionality"
        assert (
            self.pos_embeds.shape[1] == self.clip_n_dims
        ), "Embedding dimensionality must match the model dimensionality"

    @property
    def name(self) -> str:
        return "openclip_{}_{}".format(self.config.clip_model_type, self.config.clip_model_pretrained)

    @property
    def embedding_dim(self) -> int:
        return self.config.clip_n_dims
    
    def gui_cb(self,element):
        self.set_positives(element.value.split(";"))

    def set_positives(self, text_list):
        self.positives = text_list
        with torch.no_grad():
            tok_phrases = torch.cat([self.tokenizer(phrase) for phrase in self.positives]).to("cuda")
            self.pos_embeds = self.model.encode_text(tok_phrases)
        self.pos_embeds /= self.pos_embeds.norm(dim=-1, keepdim=True)

    def get_relevancy(self, embed: torch.Tensor, positive_id: int) -> torch.Tensor:
        phrases_embeds = torch.cat([self.pos_embeds, self.neg_embeds], dim=0)
        p = phrases_embeds.to(embed.dtype)  # phrases x 512
        output = torch.mm(embed, p.T)  # rays x phrases
        positive_vals = output[..., positive_id : positive_id + 1]  # rays x 1
        negative_vals = output[..., len(self.positives) :]  # rays x N_phrase
        repeated_pos = positive_vals.repeat(1, len(self.negatives))  # rays x N_phrase

        sims = torch.stack((repeated_pos, negative_vals), dim=-1)  # rays x N-phrase x 2
        softmax = torch.softmax(10 * sims, dim=-1)  # rays x n-phrase x 2
        best_id = softmax[..., 0].argmin(dim=1)  # rays x 2
        return torch.gather(softmax, 1, best_id[..., None, None].expand(best_id.shape[0], len(self.negatives), 2))[:, 0, :]

    def encode_image(self, input):
        processed_input = self.process(input).half()
        return self.model.encode_image(processed_input)





def create(image_list, data_list, save_folder):
    assert image_list is not None, "image_list must be provided to generate features"
    embed_size=512
    seg_maps = []
    total_lengths = []
    timer = 0
    img_embeds = torch.zeros((len(image_list), 300, embed_size))
    seg_maps = torch.zeros((len(image_list), 4, *image_list[0].shape[1:])) 
    mask_generator.predictor.model.to('cuda')

    for i, img in tqdm(enumerate(image_list), desc="Embedding images", leave=False):
        timer += 1
        try:
            img_embed, seg_map = _embed_clip_sam_tiles(img.unsqueeze(0), sam_encoder)
        except:
            raise ValueError(timer)

        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        total_lengths.append(total_length)
        
        if total_length > img_embeds.shape[1]:
            pad = total_length - img_embeds.shape[1]
            img_embeds = torch.cat([
                img_embeds,
                torch.zeros((len(image_list), pad, embed_size))
            ], dim=1)

        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length
        img_embeds[i, :total_length] = img_embed
        
        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]
        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map = torch.stack(seg_map_tensor, dim=0)
        seg_maps[i] = seg_map

    mask_generator.predictor.model.to('cpu')
        
    for i in range(img_embeds.shape[0]):
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        assert total_lengths[i] == int(seg_maps[i].max() + 1)
        curr = {
            'feature': img_embeds[i, :total_lengths[i]],
            'seg_maps': seg_maps[i]
        }
        sava_numpy(save_path, curr)

def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())

def _embed_clip_sam_tiles(image, sam_encoder):
    aug_imgs = torch.cat([image])
    seg_images, seg_map = sam_encoder(aug_imgs)

    clip_embeds = {}
    # 固定四个尺度，缺失的用 None 占位
    for mode in ['default', 's', 'm', 'l']:
        tiles = seg_images.get(mode, None)
        if tiles is None:
            clip_embeds[mode] = None
            continue
        tiles = tiles.to("cuda")
        with torch.no_grad():
            clip_embed = model.encode_image(tiles)
        clip_embed /= clip_embed.norm(dim=-1, keepdim=True)
        # 与原版 preprocess 行为一致：先在 GPU 用 fp16，加速/省显存
        clip_embeds[mode] = clip_embed.detach().cpu().half()

    return clip_embeds, seg_map

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """向量化实现的 mask NMS，去掉 Python 双重 for 循环。"""

    # 按得分从高到低排序
    scores, idx = scores.sort(0, descending=True)
    masks_ord = masks[idx.view(-1), :]
    num_masks = masks_ord.shape[0]

    # 展平为 (N, HW)
    n, h, w = masks_ord.shape
    masks_flat = masks_ord.reshape(n, -1).to(torch.float32)

    # 每个 mask 的面积
    masks_area = masks_flat.sum(dim=1)  # (N,)

    # 交集： (N,HW) @ (HW,N) -> (N,N)
    intersection = (masks_flat @ masks_flat.t())

    # 并集：area_i + area_j - intersection
    area_i = masks_area.view(-1, 1)
    area_j = masks_area.view(1, -1)
    union = area_i + area_j - intersection
    # 防止除零
    eps = 1e-6
    iou_matrix = intersection / (union + eps)

    # 只保留上三角，避免重复计算 + 去掉对角线
    iou_matrix = torch.triu(iou_matrix, diagonal=1)

    # inner-iou 计算：需要 intersection/area_i 和 intersection/area_j
    inter_over_i = intersection / (area_i + eps)
    inter_over_j = intersection / (area_j + eps)

    inner_iou_matrix = torch.zeros_like(iou_matrix)

    # 条件1： intersection/area_i < 0.5 且 intersection/area_j >= 0.85，对应原来的 i,j
    cond1 = (inter_over_i < 0.5) & (inter_over_j >= 0.85)
    inner_iou_matrix[cond1] = 1 - (inter_over_j[cond1] * inter_over_i[cond1])

    # 条件2：intersection/area_i >= 0.85 且 intersection/area_j < 0.5，对应原来的 j,i
    cond2 = (inter_over_i >= 0.85) & (inter_over_j < 0.5)
    # 这里需要写入到 (j,i)，所以转置索引
    inner_iou_matrix = inner_iou_matrix + inner_iou_matrix.t()
    tmp = torch.zeros_like(inner_iou_matrix)
    tmp[cond2] = 1 - (inter_over_j[cond2] * inter_over_i[cond2])
    inner_iou_matrix = inner_iou_matrix + tmp + tmp.t()

    # 取每个 mask 与其它 mask 的最大 IoU / inner-IoU
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)

    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr

    # 若没有任何满足条件的 mask，则保留 top-k（默认为 3）
    topk = min(3, num_masks)
    if keep_conf.sum() == 0:
        index = scores.topk(topk).indices
        keep_conf[index] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(topk).indices
        keep_inner_u[index] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(topk).indices
        keep_inner_l[index] = True

    keep = keep & keep_conf & keep_inner_u & keep_inner_l
    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image):
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    
    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image)
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i
        if len(seg_img_list) == 0:
            return None, seg_map
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    for name, masks in zip(['default', 's', 'm', 'l'], [masks_default, masks_s, masks_m, masks_l]):
        seg_imgs, seg_map = mask2segmap(masks, image)
        seg_images[name] = seg_imgs  # 可能为 None，占位
        seg_maps[name] = seg_map

    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="ckpts/sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path
    img_folder = os.path.join(dataset_path, 'images')
    data_list = os.listdir(img_folder)
    data_list.sort()

    # 初始化 OpenCLIP 与 SAM，仅加载一次并常驻 GPU
    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        # 适当降低采样密度以减少 mask 数量并加速
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    save_folder = os.path.join(dataset_path, 'language_features')
    os.makedirs(save_folder, exist_ok=True)

    WARNED = False
    for idx, data_path in enumerate(tqdm(data_list, desc="Embedding images", leave=False)):
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)
        if image is None:
            continue

        orig_w, orig_h = image.shape[1], image.shape[0]
        if args.resolution == -1:
            if orig_h > 1080:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_h / 1080
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down)
        resolution = (int(orig_w / scale), int(orig_h / scale))

        image_resized = cv2.resize(image, resolution)
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1)[None, ...]

        # 调用 SAM + CLIP 生成当前图片的特征与分割
        img_embed_dict, seg_map_dict = _embed_clip_sam_tiles(image_tensor, sam_encoder)

        modes = ['default', 's', 'm', 'l']

        # 计算每个尺度的长度（None 视为 0）
        lengths = []
        for m in modes:
            feat = img_embed_dict[m]
            lengths.append(0 if feat is None else len(feat))
        total_length = sum(lengths)

        if total_length == 0:
            print(f"[ WARN ] No valid masks for {data_path}, skip.")
            continue

        # 拼接不同 scale 的特征到 float32
        feats = []
        for m in modes:
            feat = img_embed_dict[m]
            if feat is None:
                continue
            feats.append(feat.float())
        img_embed = torch.cat(feats, dim=0)
        assert img_embed.shape[0] == total_length

        # 构造 seg_maps: [4, H, W]，保持 mode 顺序并修正 index 偏移
        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths_cumsum)):
            lengths_cumsum[j] += lengths_cumsum[j - 1]

        for j, m in enumerate(modes):
            v = seg_map_dict[m]
            v = v.copy()
            if lengths[j] == 0:
                # 该尺度没有特征，保持全 -1 占位
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
            else:
                offset = lengths_cumsum[j - 1]
                v[v != -1] += offset
                seg_map_tensor.append(torch.from_numpy(v))

        seg_maps = torch.stack(seg_map_tensor, dim=0)  # [4,H,W]

        save_path = os.path.join(save_folder, data_path.split('.')[0])
        assert total_length == int(seg_maps.max() + 1)

        curr = {
            'feature': img_embed,
            'seg_maps': seg_maps
        }
        sava_numpy(save_path, curr)
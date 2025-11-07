#    Copyright 2024
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
from abc import ABC, abstractmethod
import re
import copy
import torch
import torch.nn as nn
import random
import json
from typing import List, Optional, Tuple, Union, Dict

from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers import Qwen2Config

from .vision_tower_builder import build_vision_tower
from .mm_projector_builder import build_vision_projector

from .constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN
from .conversation import conv_templates, SeparatorStyle
from .mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, get_anyres_image_grid_shape, load_video
from .modeling_qwen2_flash import Qwen2Model_Flash, Qwen2ForCausalLM_Flash
from multiprocessing import Pool, cpu_count


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            delay_load = getattr(config, "delay_load", False)
            self.vision_tower = build_vision_tower(config, delay_load=delay_load)
            self.mm_projector = build_vision_projector(config, vision_cfg=self.vision_tower.config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))
            if "nopad" in getattr(config, "mm_patch_merge_type", "") and getattr(self.config, "mm_newline_position", "nothing") != "nothing":
                self.frame_newline = nn.Parameter(torch.empty(config.hidden_size, dtype=self.dtype))

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.vision_tower_pretrained = getattr(model_args, "vision_tower_pretrained", "")

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()



        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, "mm_projector_type", "linear")
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config, vision_cfg=vision_tower.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
            if "nopad" in getattr(self.config, "mm_patch_merge_type", "") and getattr(self.config, "mm_newline_position", "nothing") != "nothing":
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.frame_newline = nn.Parameter(torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location="cpu")

            def get_w(weights, keyword):
                return {k.split(keyword + ".")[1]: v for k, v in weights.items() if keyword in k}

            if self.config.mm_projector_type =='lxh_qformer':
                incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"), strict=False)
            else:
                incompatible_keys = self.mm_projector.load_state_dict(get_w(mm_projector_weights, "mm_projector"))
            print(f"Loaded mm projector weights from {pretrain_mm_mlp_adapter}. Incompatible keys: {incompatible_keys}")


# ----------------- Modified -----------------
class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def get_4dPool(self, image_feature):
        num_frames, num_tokens, num_dim = image_feature.shape
        height = width = int(math.sqrt(num_tokens))
        assert num_tokens == height * width, image_feature.shape
        
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            raise NotImplementedError
            image_feature = nn.functional.avg_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        elif self.config.mm_spatial_pool_mode == "max":
            raise NotImplementedError
            image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, weight = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / 4), math.ceil(weight / 4)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature

    def get_2dPool(self, image_feature):
        num_frames, num_tokens, num_dim = image_feature.shape
        height = width = int(math.sqrt(num_tokens))
        assert num_tokens == height * width, image_feature.shape
        
        image_feature = image_feature.view(num_frames, height, width, -1)
        image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
        # image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        if self.config.mm_spatial_pool_mode == "average":
            raise NotImplementedError
            image_feature = nn.functional.avg_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        elif self.config.mm_spatial_pool_mode == "max":
            raise NotImplementedError
            image_feature = nn.functional.max_pool2d(image_feature, self.config.mm_spatial_pool_stride)
        elif self.config.mm_spatial_pool_mode == "bilinear":
            height, weight = image_feature.shape[2:]
            scaled_shape = [math.ceil(height / 2), math.ceil(weight / 2)]
            image_feature = nn.functional.interpolate(image_feature, size=scaled_shape, mode='bilinear')

        else:
            raise ValueError(f"Unexpected mm_spatial_pool_mode: {self.config.mm_spatial_pool_mode}")
        image_feature = image_feature.permute(0, 2, 3, 1)
        image_feature = image_feature.view(num_frames, -1, num_dim)
        return image_feature


    def encode_image(self, images_list):
        concat_images = torch.cat([image for image in images_list], dim=0)
        split_sizes = [image.shape[0] for image in images_list] 

        image_features = self.get_model().get_vision_tower()(concat_images)
        image_features = self.get_model().mm_projector(image_features)
        image_features = torch.split(image_features, split_sizes)

        return image_features
    
    def encode_image_video(self, images_list, video_idx_in_batch):
        concat_images = torch.cat([image for image in images_list], dim=0)
        split_sizes = [image.shape[0] for image in images_list] 

        videos_or_images_features = self.get_model().get_vision_tower()(concat_images)
        
        per_videos_or_images_features = torch.split(videos_or_images_features, split_sizes, dim=0)  # tuple, (dim_1, 576, 4096)
        all_videos_or_images_features = []


        for idx, feat in enumerate(per_videos_or_images_features):

            if idx in video_idx_in_batch:

                feat = self.get_model().mm_projector(feat, compress=True, local_num_frames=getattr(self.config, "mm_local_num_frames", -1))
            else:

                feat = self.get_model().mm_projector(feat, compress=False)

            all_videos_or_images_features.append(feat)


        return all_videos_or_images_features


    def encode_video(self, images_list, video_idx_in_batch):

        bs = len(images_list)

        concat_images = []
        concat_videos = []
        for idx, image in enumerate(images_list):
            if idx in video_idx_in_batch:
                concat_videos.append(image)
            else:
                concat_images.append(image)
        # print(concat_videos[0].shape)
        has_image = len(concat_images) > 0
        has_video = len(concat_videos) > 0

        mm_local_num_frames = getattr(self.config, "mm_local_num_frames", -1)
        assert mm_local_num_frames != -1
        if has_image:
            image_split_sizes = [image.shape[0] for image in concat_images] 
            concat_images = torch.cat([image.unsqueeze(1) for image in concat_images], dim=0)
            images_features = self.get_model().get_vision_tower()(concat_images) # B_i, N, D
            images_features = self.get_model().mm_projector(images_features, compress=False, local_num_frames=1)
            images_features = torch.split(images_features, image_split_sizes)

        if has_video:
            video_split_sizes = [video.shape[0] // mm_local_num_frames for video in concat_videos]
            concat_videos = torch.cat([video.reshape(video.shape[0] // mm_local_num_frames, mm_local_num_frames, video.shape[1], video.shape[2], video.shape[3]) for video in concat_videos], dim=0) #  B T C H W
            videos_features = self.get_model().get_vision_tower()(concat_videos) # B_v, N, D
            videos_features = self.get_model().mm_projector(videos_features, compress=True, local_num_frames=mm_local_num_frames)
            videos_features = [v.reshape(-1, v.shape[-2] // mm_local_num_frames, v.shape[-1]) for v in torch.split(videos_features, video_split_sizes)]


        all_videos_or_images_features = []
        img_idx = 0
        vid_idx = 0

        for idx in range(bs):
            
            if idx in video_idx_in_batch:
                feat =videos_features[vid_idx]
                vid_idx += 1
            else:
                feat = images_features[img_idx]
                img_idx += 1

            all_videos_or_images_features.append(feat)

        if has_video:
            assert vid_idx == len(videos_features), f"vid: {vid_idx} != {len(videos_features)}"
        if has_image:
            assert img_idx == len(images_features), f"img: {img_idx} != {len(images_features)}"

        return all_videos_or_images_features

    def encode_video_image(self, images_list, video_idx_in_batch):

        bs = len(images_list)

        concat_images = []
        concat_videos = []
        for idx, image in enumerate(images_list):
            if idx in video_idx_in_batch:
                concat_videos.append(image)
            else:
                concat_images.append(image)
        # print(concat_videos[0].shape)
        has_image = len(concat_images) > 0
        has_video = len(concat_videos) > 0

        mm_local_num_frames = getattr(self.config, "mm_local_num_frames", -1)
        assert mm_local_num_frames != -1
        if has_image:
            image_split_sizes = [image.shape[0] for image in concat_images] 
            concat_images = torch.cat([image.unsqueeze(1) for image in concat_images], dim=0)
            # print("input vit image.shape:", concat_images.shape)
            images_features = self.get_model().get_vision_tower()(concat_images) # B_i, N, D
            images_features = torch.split(images_features, image_split_sizes)

        if has_video:
            video_split_sizes = [video.shape[0] // mm_local_num_frames for video in concat_videos]
            concat_videos = torch.cat([video.reshape(video.shape[0] // mm_local_num_frames, mm_local_num_frames, video.shape[1], video.shape[2], video.shape[3]) for video in concat_videos], dim=0)
            # print("input vit video.shape:", concat_videos.shape)
            videos_features = self.get_model().get_vision_tower()(concat_videos) # B_v, N, D
            videos_features = [v.reshape(-1, v.shape[-2] // mm_local_num_frames, v.shape[-1]) for v in torch.split(videos_features, video_split_sizes)]


        all_videos_or_images_features = []
        img_idx = 0
        vid_idx = 0

        for idx in range(bs):
            
            if idx in video_idx_in_batch:
                feat = self.get_model().mm_projector(videos_features[vid_idx], compress=True, local_num_frames=getattr(self.config, "mm_local_num_frames", -1), condenser=self.condenser)
                
                vid_idx += 1
            else:
                # feat = self.get_model().mm_projector(images_features[img_idx], compress=False)
                feat = self.get_model().mm_projector(images_features[img_idx], compress=True, local_num_frames=1, condenser=self.condenser)
                img_idx += 1

            all_videos_or_images_features.append(feat)

        if has_video:
            assert vid_idx == len(videos_features), f"vid: {vid_idx} != {len(videos_features)}"
        if has_image:
            assert img_idx == len(images_features), f"img: {img_idx} != {len(images_features)}"

        return all_videos_or_images_features

    def add_token_per_frame(self, image_feature):
        image_feature = image_feature.permute(2, 0, 1).contiguous()
        if hasattr(self.model, "frame_newline"):
            image_feature =  torch.cat((image_feature, self.model.frame_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        else:
            image_feature =  torch.cat((image_feature, self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)), dim=-1)
        image_feature = image_feature.permute(1, 2, 0).contiguous()
        return image_feature
    
    def add_different_token_per_frame(self, image_feature):
        raise NotImplementedError("No")

        
    def prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities=["image"], image_sizes=None):
        assert type(modalities) is list, modalities
        
        vision_tower = self.get_vision_tower()
        # rank_print(modalities)
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]

            video_idx_in_batch = []
            for _ in range(len(modalities)):
                if modalities[_] == "video":
                    video_idx_in_batch.append(_)

            images_list = []
            for image in images:
                if image.ndim == 4:
                    images_list.append(image)
                else:
                    images_list.append(image.unsqueeze(0))


            vision_encode_type = getattr(self.config, "vision_encode_type", "image")
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            frame_aspect_ratio = getattr(self.config, "frame_aspect_ratio", "square")
            mm_newline_position = getattr(self.config, "mm_newline_position", "nothing")

            if "anyres" in frame_aspect_ratio:
                new_images_list = []
                num_frames_list = []
                for idx, image in enumerate(images_list):
                    if idx in video_idx_in_batch:
                        T, C, H, W = image.shape
                        num_frames_list.append(T)
                        # print("origin video.shape:", image.shape) # T C H W
                        patch_size = self.get_vision_tower().image_size


                        if H * W != patch_size * patch_size:
                            global_image = F.interpolate(
                                image.float(), size=(patch_size, patch_size), mode='bicubic', align_corners=False
                            ).to(image.dtype).unsqueeze(0)
                            sub_image = image.reshape(
                                1, T, C, H//patch_size, patch_size, W//patch_size, patch_size
                            ).permute(0, 3, 5, 1, 2, 4, 6).reshape(-1, T, C, patch_size, patch_size).contiguous()
                            new_image = torch.concat([global_image, sub_image], dim=0).flatten(0, 1)
                        else:
                            new_image = image

                        # print("new video shape:", new_image.shape)
                        new_images_list.append(new_image)
                    else:
                        num_frames_list.append(1)
                        new_images_list.append(image)

                images_list = new_images_list


            # rank0_print(self.config)
            # TODO image: share vit&connector for image/video, image_video:, video
            if vision_encode_type == "image": # image backbone, process video by frame
                image_features = self.encode_image(images_list)
            elif vision_encode_type == "video": # video backbone, process video with compress
                image_features = self.encode_video(images_list, video_idx_in_batch=video_idx_in_batch)
            elif vision_encode_type == "image_video": # image backbone, process video with compress
                image_features = self.encode_image_video(images_list, video_idx_in_batch=video_idx_in_batch)
            elif vision_encode_type == "image_video_new":
                image_features = self.encode_image_video_new(images_list, video_idx_in_batch=video_idx_in_batch)
            elif vision_encode_type == "video_image": # image backbone, process video with compress
                # image_features = self.encode_video_image(images_list, video_idx_in_batch=video_idx_in_batch)
                torch.cuda.empty_cache()
                assert len(images_list) == 1, f"Only support single image/video: {images_list}"
                if len(video_idx_in_batch) > 0 and images_list[0].shape[0] > 512: # that means video_idx_in_batch == [0]
                    if images_list[0].shape[0] < 1024:
                        n_frames = images_list[0].shape[0]
                        first_half = n_frames // 2 // 8 * 8
                        with torch.no_grad():
                            images_features_1 = self.encode_video_image([images_list[0][:first_half]], video_idx_in_batch=video_idx_in_batch)
                        torch.cuda.empty_cache()
                        with torch.no_grad():
                            images_features_2 = self.encode_video_image([images_list[0][first_half:]], video_idx_in_batch=video_idx_in_batch)
                        torch.cuda.empty_cache()
                        image_features = [torch.cat((images_features_1[0], images_features_2[0]), dim=0)]
                    else:
                        n_segments = images_list[0].shape[0] // 256
                        n_locals = images_list[0].shape[0] // 8
                        image_features = []
                        for i in range(n_segments):
                            start = n_locals * i // n_segments * 8
                            end = n_locals * (i + 1) // n_segments * 8
                            with torch.no_grad():
                                image_features.append(self.encode_video_image([images_list[0][start:end]], video_idx_in_batch=video_idx_in_batch)[0])
                            torch.cuda.empty_cache()
                        image_features = [torch.cat(image_features, dim=0)]
                else:
                    with torch.no_grad():
                        image_features = self.encode_video_image(images_list, video_idx_in_batch=video_idx_in_batch)
                torch.cuda.empty_cache()
            else:
                raise NotImplementedError(vision_encode_type)
            

            if 'llava_ov' in getattr(self.config, "transformers_version", ""):
                new_image_features = []
                # print("I am llava ov!!!!!!!")
                for idx, image_feat in enumerate(image_features):
                    if idx in video_idx_in_batch: # NOTE lxh: I don't want it.
                        new_image_features.append(self.get_2dPool(image_feat))
                    else:
                        new_image_features.append(image_feat)
                image_features = new_image_features
                
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    # FIXME: now assume the image is square, and split to 2x2 patches
                    # num_patches = h * w, where h = w = sqrt(num_patches)
                    # currently image_feature is a tensor of shape (4, num_patches, hidden_size)
                    # we want to first unflatten it to (2, 2, h, w, hidden_size)
                    # rank0_print("At least we are reaching here")
                    if image_idx in video_idx_in_batch:  # video operations
                        # rank0_print("Video")
                        # rank0_print(f"video image_feature.shape: {image_feature.shape}")

                        if "anyres" in frame_aspect_ratio:
                            if "anyres_max" in frame_aspect_ratio:
                                matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", frame_aspect_ratio)
                                if matched_anyres_max_num_patches:
                                    max_num_patches = int(matched_anyres_max_num_patches.group(1))
                            
                            num_frames = num_frames_list[image_idx]
                            
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.frame_grid_pinpoints, vision_tower_image_size, max_resolutions=self.config.max_num_pixels // num_frames) #TODO 要传个num_frames来算
                            except Exception as e:
                                rank0_print(f"Error: {e}, self.config:{self.config}")
                                raise e
                                
                            height = width = self.get_model().mm_projector.num_frame_patches_per_side


                            if "maxpool2x2" in mm_patch_merge_type:
                                raise NotImplementedError
                            elif "unpad" in mm_patch_merge_type and "anyres_max" in frame_aspect_ratio and matched_anyres_max_num_patches:
                                raise NotImplementedError
                            elif "unpad" in mm_patch_merge_type and "anyres" in frame_aspect_ratio:
                                raise NotImplementedError
                            else:
                                # rank0_print(f"652 num_frames={num_frames}")

                                if num_patch_height * num_patch_width != 1: # has global
                                    image_feature = image_feature.view(num_patch_height * num_patch_width + 1, -1,  height, width, image_feature.shape[-1])
                                    assert num_frames == image_feature.shape[1], f"{num_frames} != {image_feature.shape[1]}"
                                    
                                    base_frame_feature = image_feature[0].view(num_frames, -1, image_feature[0].shape[-1]) # T 4*4 C
                                    # rank0_print(f"base_frame_feature.shape: {base_frame_feature.shape}")
                                    image_feature = image_feature[1:].permute(1, 0, 2, 3, 4) # T P 4 4 C
                                    frame_feature = image_feature.view(num_frames, num_patch_height, num_patch_width, height, width, -1)
                                    frame_feature = frame_feature.permute(0, 1, 3, 2, 4, 5).contiguous()
                                    frame_feature = frame_feature.flatten(1, 4)
                                    frame_feature = torch.cat((base_frame_feature, frame_feature), dim=1)
                                    # rank0_print(f"two_frame_feature.shape: {frame_feature.shape}")
                                else: # no global
                                    frame_feature = image_feature.view(num_frames, -1, image_feature[0].shape[-1]) # T 4*4 C
                                    # rank0_print(f"only_frame_feature.shape: {frame_feature.shape}")

                            if "nobase" in mm_patch_merge_type:
                                raise NotImplementedError

                        else:
                            frame_feature = image_feature

                        if "pad" in mm_patch_merge_type: # unpad和nopad都算
                            if mm_newline_position == 'one_token':
                                frame_feature = frame_feature.flatten(0, 1)
                                if "unpad" in mm_patch_merge_type:
                                    frame_feature = torch.cat((frame_feature, self.model.image_newline[None].to(frame_feature.device)), dim=0)
                                else:
                                    frame_feature = torch.cat((frame_feature, self.model.frame_newline[None].to(frame_feature.device)), dim=0)
                            elif mm_newline_position == 'frame':
                                # Frame-wise
                                frame_feature = self.add_token_per_frame(frame_feature)
                                frame_feature = frame_feature.flatten(0, 1)
                            elif mm_newline_position == 'frame2':
                                # Frame-wise
                                raise NotImplementedError
                            elif mm_newline_position == 'nothing':
                                frame_feature = frame_feature.flatten(0, 1)
                            else:
                                raise NotImplementedError("add pad please!!")
                        else:
                            frame_feature = frame_feature.flatten(0, 1)

                        # rank0_print(f"final video frame_feature.shape: {frame_feature.shape}")
                        image_feature = frame_feature

                    elif image_feature.shape[0] > 1:  # multi patches and multi images operations
                        # rank0_print("Single-images") NOTE: 多图实际上不会过这里，因为被拆成多个单图pad了
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]

                        origin_size = image_feature.shape
                        
                    
                        height = width = self.get_model().mm_projector.num_image_patches_per_side # NOTE 写死一个图49
                        assert height * width == base_image_feature.shape[0], f"height:{height}, width: {width}, base_image_feature: {base_image_feature.shape}"

                        if "anyres_max" in image_aspect_ratio:
                            matched_anyres_max_num_patches = re.match(r"anyres_max_(\d+)", image_aspect_ratio)
                            if matched_anyres_max_num_patches:
                                max_num_patches = int(matched_anyres_max_num_patches.group(1))

                        if "anyres" in image_aspect_ratio:
                            if hasattr(self.get_vision_tower(), "image_size"):
                                vision_tower_image_size = self.get_vision_tower().image_size
                            else:
                                raise ValueError("vision_tower_image_size is not found in the vision tower.")
                            try:
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, vision_tower_image_size, max_resolutions=None) 
                            except Exception as e:
                                rank0_print(f"Error: {e}")
                                raise e
                                # num_patch_width, num_patch_height = 2, 2

                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError(image_aspect_ratio)
                            image_feature = image_feature.view(2, 2, height, width, -1)

                        if "maxpool2x2" in mm_patch_merge_type:
                            raise NotImplementedError
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = nn.functional.max_pool2d(image_feature, 2)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        elif "unpad" in mm_patch_merge_type and "anyres_max" in image_aspect_ratio and matched_anyres_max_num_patches:
                            raise NotImplementedError
                        elif "unpad" in mm_patch_merge_type:
                            raise NotImplementedError
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        if "nobase" in mm_patch_merge_type:
                            pass
                        else:
                            try:
                            
                                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                            except Exception as e:
                                raise ValueError(f"{num_patch_width} {num_patch_height} now: base_image_feature: {base_image_feature.shape}, {image_feature.shape}, image_sizes[image_idx]: {image_sizes[image_idx]}, origin_size: {origin_size}, {image_sizes[image_idx]}, {self.config.image_grid_pinpoints}, {vision_tower_image_size}")
                    else:  # single image operations
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat((image_feature, self.model.image_newline[None]), dim=0)

                    # rank0_print(f"image/video_feature.shape: {image_feature.shape}")
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            # raise NotImplementedError(f"images.shape={images.shape},  modalities={modalities}")
            image_features = self.encode_image(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(self.config, "mm_use_im_start_end", False):
            raise NotImplementedError
        # rank0_print(f"Total images len(image_features: {len(image_features)}")

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        # mm_llm_compress = getattr(self.config, "mm_llm_compress", False)
        mm_llm_compress = True
        
        if mm_llm_compress:
            # self.model.llm_compress_type = getattr(self.config, "llm_compress_type", "attention")
            self.model.llm_compress_type = "uniform"
            # self.model.llm_compress_layer_list = getattr(self.config, "llm_compress_layer_list", [8, 16, 24])
            self.model.llm_compress_layer_list = eval(os.environ.get("EXTRA_PARAM_INNER_CONDENSER_ID", "[]"))
            # self.model.llm_image_token_ratio_list = getattr(self.config, "llm_image_token_ratio_list", [1.0, 0.5, 0.25, 0.125])
            if 'EXTRA_PARAM_OVERWRITE_DROP_SCHEDULE' in os.environ:
                len_of_each = [len(o) for o in self.idx_lists]
                self.model.llm_image_token_ratio_list = [1.0] + [len_of_each[_+1] / 16 for _ in self.model.llm_compress_layer_list]
            else:
                self.model.llm_image_token_ratio_list = [1.]
                inner_stride = float(os.environ.get("EXTRA_PARAM_INNER_STRIDE", 1))
                inner_stride_list = [inner_stride] * len(self.model.llm_compress_layer_list)
                if 'EXTRA_PARAM_INNER_STRIDE_LIST' in os.environ:
                    inner_stride_list = eval(os.environ.get("EXTRA_PARAM_INNER_STRIDE_LIST", "[]"))
                for in_st in inner_stride_list:
                    self.model.llm_image_token_ratio_list.append(self.model.llm_image_token_ratio_list[-1] / in_st)
            if 'EXTRA_PARAM_ATTENTION_DROP' in os.environ:
                self.model.target_n_clips = float(os.environ.get("EXTRA_PARAM_ATTENTION_DROP", 0.5))
                self.model.clip_ratio_list = [1.0]
                n_drop_clip = len(self.model.llm_compress_layer_list)
                for i in range(n_drop_clip):
                    self.model.clip_ratio_list.append(1 - 0.5*(i+1)/n_drop_clip)
            else:
                self.model.target_n_clips = 1
                self.model.clip_ratio_list = [1.0] * 100
            first_image_token_position = []
            text_prompt_lens = []
        else:
            assert False, "Ciallo~ (∠・ω< )⌒☆"
            self.model.llm_compress_type = "attention"
            self.model.llm_compress_layer_list = []
            self.model.llm_image_token_ratio_list = []
            
        # rank_print("Inserting Images embedding")
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()

            if mm_llm_compress:
                ####### copy from pdrop, only support single image/video NOTE ##################
                # record image position for further dropping
                image_index = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                assert len(image_index) == 1, f"Only support singe/video: {image_index}"
                if image_index == []:
                    first_image_token_position.append(-1)
                else:
                    first_image_token_position.append(image_index[0])
                

                # record input instruction length in inference mode
                if not self.training:  
                    if image_index == []:
                        assert num_images == 0, num_images
                    else:
                        assert num_images == 1, f"num_images={num_images}, not support"
                    text_prompt_lens.append(cur_input_ids.shape[0] - num_images)   # consider image place holder

                ###############################################


            # rank0_print(f"num_images={num_images}")
            if num_images == 0: 
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i] + 1 : image_token_indices[i + 1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    try:
                        cur_image_features = image_features[cur_image_idx]
                    except IndexError:
                        rank0_print(f"cur_image_idx={cur_image_idx} is not ok")
                        cur_image_features = image_features[cur_image_idx - 1]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            # import pdb; pdb.set_trace()
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)


        if mm_llm_compress:
            self.model.first_image_token_position = first_image_token_position 
            self.model.text_prompt_lens = text_prompt_lens
            self.model.num_image_token_lens = [image_feature.shape[0] for image_feature in image_features]
            self.model.num_clips = [x//128 for x in self.model.num_image_token_lens]
        
        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, "tokenizer_model_max_length", None)
        # rank_print("Finishing Inserting")

        new_input_embeds = [x[:tokenizer_model_max_length] for x, modality in zip(new_input_embeds, modalities)]
        new_labels = [x[:tokenizer_model_max_length] for x, modality in zip(new_labels, modalities)]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
        # rank0_print("Prepare pos id")

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(torch.cat((torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device), cur_new_embed), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((cur_new_embed, torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # rank0_print("tokenizer padding")

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None
        if getattr(self.config, "use_pos_skipping", False) and self.training:
            position_ids = torch.arange(new_input_embeds.size(1), device=new_input_embeds.device).unsqueeze(0).to(new_input_embeds.device)
            split_position = random.randint(0, new_input_embeds.size(1))
            left_add = random.randint(0, self.config.pos_skipping_range)
            right_add = random.randint(left_add, self.config.pos_skipping_range)
            position_ids[:, :split_position] += left_add
            position_ids[:, split_position:] += right_add
        # import pdb; pdb.set_trace()
        # print("Finish preparing")

        # print("Input shape:", new_input_embeds.shape)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location="cpu")
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
# ----------------- End Modified -----------------


class VideoChatFlashQwenConfig(Qwen2Config):
    model_type = "videochat_flash_qwen"


class VideoChatFlashQwenModel(LlavaMetaModel, Qwen2Model_Flash):
    config_class = VideoChatFlashQwenConfig

    def __init__(self, config: VideoChatFlashQwenConfig):
        super(VideoChatFlashQwenModel, self).__init__(config)


def preprocess_frame(args):
    frame, dtype, image_processor = args
    processed = image_processor.preprocess(frame, return_tensors="pt")["pixel_values"]
    return processed.to(dtype).cuda()

class VideoChatFlashQwenForCausalLM(LlavaMetaForCausalLM, Qwen2ForCausalLM_Flash):
    config_class = VideoChatFlashQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM_Flash.__init__(self, config)
        config.model_type = "videochat_flash_qwen"
        # config.rope_scaling = None

        self.model = VideoChatFlashQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing


        # ----------------- Modified -----------------
        OUTER_CONDENSER = os.environ.get('EXTRA_PARAM_OUTER_CONDENSER_TYPE', None)
        OUTER_LAYERS = int(os.environ.get('EXTRA_PARAM_OUTER_CONDENSER_LAYER', '0'))
        INNER_CONDENSER = os.environ.get('EXTRA_PARAM_INNER_CONDENSER_TYPE', None)
        INNER_LAYERS = int(os.environ.get('EXTRA_PARAM_INNER_CONDENSER_LAYER', '0'))

        print("It is LlavaQwenForCausalLM")
        # import pdb; pdb.set_trace()
        from llava.model.condenser_arch import SelfAttentionCondenser, AvgPoolingCondenser, TransformerCondenser, RemoveFirstFrameCondenser, IdentityCondenser, SelectTheNextHalf, StupidPooling, SelectRowByRow

        if OUTER_CONDENSER is not None:
            if OUTER_CONDENSER == 'rotary':
                self.condenser = SelfAttentionCondenser(hidden_size=1024, num_layers=OUTER_LAYERS, position_embedding_type='rotary')
            else:
                assert False, "Unsupported condenser type"
        else:
            self.condenser = IdentityCondenser()

        learnable_token_stride = int(os.environ.get("EXTRA_PARAM_OUTER_LEARNABLE_TOKEN", 0))
        if learnable_token_stride > 0:
            self.condenser.learnable_token = nn.Parameter(torch.randn(1, config.hidden_size))
            self.condenser.learnable_token_stride = learnable_token_stride
        
        if INNER_CONDENSER is not None:
            inner_condense_layers = eval(os.environ.get("EXTRA_PARAM_INNER_CONDENSER_ID", "[]"))
            if INNER_CONDENSER == 'rotary':
                layer_list = [SelfAttentionCondenser(hidden_size=config.hidden_size, num_layers=INNER_LAYERS, position_embedding_type='rotary') for _ in inner_condense_layers]
            elif INNER_CONDENSER == 'avgpool':
                layer_list = [AvgPoolingCondenser(hidden_size=config.hidden_size, num_layers=INNER_LAYERS) for _ in inner_condense_layers]
            elif INNER_CONDENSER == 'rowbyrow':
                layer_list = [SelectRowByRow() for _ in inner_condense_layers]
            elif INNER_CONDENSER == 'nexthalf':
                if 'EXTRA_PARAM_OVERWRITE_DROP_SCHEDULE' in os.environ:
                    assert os.environ.get("EXTRA_PARAM_OVERWRITE_DROP_SCHEDULE", None) == "Cosine", "Only support Cosine schedule"
                    n_layers = 28
                    idx_lists = []
                    cur_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
                    idx_lists.append(copy.deepcopy(cur_list))
                    import math
                    for i in range(1, n_layers + 1):
                        # lower_bound 1, upper_bound 16, T = n_layers
                        goal_len = 1 + int(15 * 0.5 * (1 + math.cos(i / n_layers * math.pi))+0.5)
                        while goal_len < len(cur_list):
                            best_score = 1e9
                            best_next = None
                            for remove in cur_list:
                                next_list = copy.deepcopy(cur_list)
                                next_list.remove(remove)
                                score = 0
                                for j in range(len(next_list) - 1):
                                    score += ( (j+1)/len(next_list) * 16 - next_list[j] ) ** 2
                                if score < best_score:
                                    best_score = score
                                    best_next = copy.deepcopy(next_list)
                            cur_list = best_next
                        idx_lists.append(copy.deepcopy(cur_list))
                        print(i, goal_len, cur_list)
                    # import pdb; pdb.set_trace()
                    self.idx_lists = idx_lists
                    inner_condense_layers = []
                    for i in range(n_layers):
                        if len(idx_lists[i]) != len(idx_lists[i+1]):
                            inner_condense_layers.append(i)
                    print("inner_condense_layers:", inner_condense_layers)
                    os.environ["EXTRA_PARAM_INNER_CONDENSER_ID"] = str(inner_condense_layers)
                    layer_list = [SelectTheNextHalf(prev_idx=idx_lists[_], next_idx=idx_lists[_+1]) for _ in inner_condense_layers]
                else:
                    layer_list = [SelectTheNextHalf() for _ in inner_condense_layers]
            else:
                assert False, "Unsupported inner condenser type"
            self.condenser.inner = nn.ModuleList(layer_list)

        if "EXTRA_PARAM_FREEZE_LAYER" in os.environ:
            n_freeze = int(os.environ.get("EXTRA_PARAM_FREEZE_LAYER", 0))
            self.condenser.freeze_layers = copy.deepcopy(self.model.layers[:n_freeze])
        # ----------------- End Modified -----------------

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        first_conv_length=None, # !!!!!!!!!!!!!!!!!!!!!
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # !!!!!!!!!!!!!!!!!!!
        if first_conv_length is not None: # note that this can also be initialized by "generate"
            self.model.first_conv_length = first_conv_length

        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        # print("inputs_embeds.shape:", inputs_embeds.shape)
        if dpo_forward:
            raise NotImplementedError
        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                # ----------------- Modified -----------------
                condenser=self.condenser,
                # ----------------- End Modified -----------------
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        first_conv_length=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:

        # import pdb; pdb.set_trace()

        # !!!!!!!!!!!!!!!!!!!!!!!!!
        self.model.first_conv_length = first_conv_length

        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            self.model.image_token_posi = [-1]     
            self.model.prompt_len = None
            self.model.image_tokens = [0]
            inputs_embeds = self.get_model().embed_tokens(inputs)
        
        print(f'prepared imputs embeds')

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def process_frames_multiprocessing(self, frames):
        image_processor = self.get_vision_tower().image_processor
        dtype = self.model.dtype

        with Pool(16) as pool:
            processed_frames = pool.map(preprocess_frame, [(frame, dtype, image_processor) for frame in frames])
        
        return processed_frames

    @torch.no_grad()
    def old_chat(self,
        video_path,
        tokenizer,
        user_prompt,
        chat_history=None,
        return_history=True,
        max_num_frames=512,
        media_dict=None,
        generation_config={}):

        print(f"start chatting with video: {video_path}")
    
        frames, time_msg  = load_video(video_path, max_num_frames=max_num_frames, media_dict=media_dict)

        image_sizes = [frames[0].shape[:2]]

        print(f'successfully loaded video')

        frames = [self.get_vision_tower().image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self.model.dtype).cuda()]

        # frames = self.process_frames_multiprocessing(frames)

        # print(f'processed frames')


        conv = conv_templates["qwen_2"].copy()

        # !!!!!!!!!!!!!!!!!!!!!!!!!!! 这里是造 prompt 的地方
        # import pdb; pdb.set_trace()
        image_last = os.environ.get("EXTRA_PARAM_IMAGE_LAST", None)
        if image_last is not None:
            if chat_history is None or len(chat_history) == 0:
                # user_prompt = f'{user_prompt}\n{time_msg.strip()} {DEFAULT_IMAGE_TOKEN}'
                if image_last.strip().lower() == "middle":
                    # replace_token = replace_token + msg.rstrip()
                    user_prompt = f'{user_prompt} {DEFAULT_IMAGE_TOKEN}\n{time_msg.strip()} {user_prompt}'
                elif image_last.strip().lower() == "true":
                    # replace_token = msg.rstrip() + " " + replace_token
                    user_prompt = f'{user_prompt}\n{time_msg.strip()} {DEFAULT_IMAGE_TOKEN}'
                else:
                    raise ValueError(f"Unknown image_last value: {image_last}. Please set it to true or false.")
            else:
                assert False, "chat_history is not None, please check."
        else:
            if chat_history is None or len(chat_history) == 0:
                user_prompt = f'{DEFAULT_IMAGE_TOKEN}\n{time_msg.strip()} {user_prompt}'
            else:
                assert DEFAULT_IMAGE_TOKEN in chat_history[0]['content'], chat_history
                for msg in chat_history:
                    conv.append_message(msg['role'], msg['content'])
        
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        print(f'built prompt')

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        if tokenizer.pad_token_id is None:
            if "qwen" in tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                tokenizer.pad_token_id = 151643

        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        def get_first_conv_length(cur_id):
            # the first appearance of [151644, 77091], return the idx of 151644
            for idx in range(len(cur_id) - 1):
                if cur_id[idx] == 151644 and cur_id[idx + 1] == 77091:
                    return idx
            assert False, f"151644 and 77091 not found in {cur_id}"

        # import pdb; pdb.set_trace()
        first_conv_length = [get_first_conv_length(cur_id) for cur_id in input_ids]


        print('start generating')

        with torch.inference_mode():
            output_ids = self.generate(
                inputs=input_ids,
                images=frames,
                attention_mask=attention_masks,
                modalities=["video"],
                image_sizes=image_sizes,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                first_conv_length=first_conv_length,
                **generation_config
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]

        outputs = outputs.strip()

        # print(f"\033[91m== Question: \033[0m\n{prompt}\n")
        # print(f"\033[91m== Response: \033[0m\n{outputs}\n")
        
        if chat_history is None:
            chat_history = []

        chat_history.append({"role":conv.roles[0], "content":user_prompt})
        chat_history.append({"role":conv.roles[1], "content":outputs})
        if return_history:
            return outputs, chat_history
        else:
            return outputs

    @torch.no_grad()
    def chat(self,
        video_path,
        tokenizer,
        user_prompt,
        chat_history=None,
        return_history=True,
        max_num_frames=512,
        media_dict=None,
        generation_config={},
        extra_dict=None):   # extra_dict: expect {"gt_id": int, "save_path": str, "max_segments": int}

        print(f"start chatting with video: {video_path}")
    
        frames, time_msg  = load_video(video_path, max_num_frames=max_num_frames, media_dict=media_dict)

        image_sizes = [frames[0].shape[:2]]

        # print(f'successfully loaded video')

        frames = [self.get_vision_tower().image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self.model.dtype).cuda()]

        # frames = self.process_frames_multiprocessing(frames)

        if len(frames) == 1:
            print('!'*80)
            print(f"processed frames len: {frames[0].shape}")
            print('!'*80)
        else:
            print(f'processed frames')


        conv = conv_templates["qwen_2"].copy()

        # !!!!!!!!!!!!!!!!!!!!!!!!!!! 这里是造 prompt 的地方
        # import pdb; pdb.set_trace()
        image_last = os.environ.get("EXTRA_PARAM_IMAGE_LAST", None)
        if image_last is not None:
            if chat_history is None or len(chat_history) == 0:
                # user_prompt = f'{user_prompt}\n{time_msg.strip()} {DEFAULT_IMAGE_TOKEN}'
                if image_last.strip().lower() == "middle":
                    # replace_token = replace_token + msg.rstrip()
                    user_prompt = f'{user_prompt} {DEFAULT_IMAGE_TOKEN}\n{time_msg.strip()} {user_prompt}'
                elif image_last.strip().lower() == "true":
                    # replace_token = msg.rstrip() + " " + replace_token
                    user_prompt = f'{user_prompt}\n{time_msg.strip()} {DEFAULT_IMAGE_TOKEN}'
                else:
                    raise ValueError(f"Unknown image_last value: {image_last}. Please set it to true or false.")
            else:
                assert False, "chat_history is not None, please check."
        else:
            if chat_history is None or len(chat_history) == 0:
                user_prompt = f'{DEFAULT_IMAGE_TOKEN}\n{time_msg.strip()} {user_prompt}'
            else:
                assert DEFAULT_IMAGE_TOKEN in chat_history[0]['content'], chat_history
                for msg in chat_history:
                    conv.append_message(msg['role'], msg['content'])
        
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        print(f'built prompt')

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        if tokenizer.pad_token_id is None:
            if "qwen" in tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                tokenizer.pad_token_id = 151643

        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        def get_first_conv_length(cur_id):
            # the first appearance of [151644, 77091], return the idx of 151644
            for idx in range(len(cur_id) - 1):
                if cur_id[idx] == 151644 and cur_id[idx + 1] == 77091:
                    return idx
            assert False, f"151644 and 77091 not found in {cur_id}"

        # import pdb; pdb.set_trace()
        first_conv_length = [get_first_conv_length(cur_id) for cur_id in input_ids]


        print('start generating')
        
        # import pdb; pdb.set_trace()

        SELECT_AND_DROP = int(os.environ.get("EXTRA_PARAM_SELECT_AND_DROP", '2048'))

        if SELECT_AND_DROP <= 0:
            SELECT_AND_DROP = None
        
        # if frames[0].shape[0] not in [1024, 2048, 3072, 4096]:
        #     SELECT_AND_DROP = None
        if frames[0].shape[0] < 1024:
            SELECT_AND_DROP = None

        if SELECT_AND_DROP is not None:
            max_segments = 512
            stride = 512
            min_rounds = (frames[0].shape[0] + stride - 1) // stride
            score_type = os.environ.get("EXTRA_PARAM_SCORE_TYPE", "default")
            if score_type.startswith("segments"):
                a = int(score_type.split("_")[1])
                b = int(score_type.split("_")[2])
                assert a%b==0, f"score_type: {score_type} is not valid"
                max_segments += b*8*2

            duplic = int(os.environ.get("EXTRA_PARAM_DUPLICATE", "8"))
            output_json = {}
            for i in range(duplic * min_rounds):
                start = (i * stride + i//min_rounds*(8//duplic) + i//min_rounds*(stride//duplic)) % frames[0].shape[0]
                end = (start + max_segments) % frames[0].shape[0]
                if start < end:
                    cur_segment = [frames[0][start:end]]
                else:
                    cur_segment = [torch.cat([frames[0][start:], frames[0][:end]])]
                with torch.inference_mode():
                    output_ids = self.generate(
                        inputs=input_ids,
                        images=cur_segment,
                        attention_mask=attention_masks,
                        modalities=["video"],
                        image_sizes=image_sizes,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                        first_conv_length=first_conv_length,
                        **generation_config
                    )
                output_json[f'segment_{i}'] = {
                    'start': start,
                    'end': end,
                    'score': self.model.clip_scores_cache,
                }
                self.model.clip_scores_cache = {}
                torch.cuda.empty_cache()
            
            scores = []
            for l in range(28): # n_layers
                tmp = [[] for _ in range(len(frames[0]))]
                scores.append(tmp)
            
            for cur_seg_id in range(duplic * min_rounds):
                cur_segment = output_json[f'segment_{cur_seg_id}']
                p = cur_segment['start']
                for l in range(28):
                    for i in range(max_segments):
                        cur_i = (i + p) % frames[0].shape[0]
                        value = cur_segment['score'][l][i//8]
                        if value == value and value < 1000:
                            scores[l][cur_i].append(value)
            
            sum_score = [0] * frames[0].shape[0]
            for i in range(6, 24):
                for j in range(frames[0].shape[0]):
                    sum_score[j] += sum(scores[i][j]) / len(scores[i][j])
            
        
        if SELECT_AND_DROP is None:
            new_frames = frames
        else:
            sum_score_w_idx = [(i, sum_score[i]) for i in range(len(sum_score))]
            sum_score_w_idx.sort(key=lambda x: x[1], reverse=True)
            selected_idx = [x[0] for x in sum_score_w_idx[:SELECT_AND_DROP]]
            selected_idx.sort()
            # frames[0] is a torch tensor
            print(f'selected idx: {selected_idx}')
            new_frames = [frames[0][selected_idx]]

        with torch.inference_mode():
            output_ids = self.generate(
                inputs=input_ids,
                images=new_frames,
                attention_mask=attention_masks,
                modalities=["video"],
                image_sizes=image_sizes,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                first_conv_length=first_conv_length,
                **generation_config
            )


        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if outputs.endswith(stop_str):
            outputs = outputs[: -len(stop_str)]

        outputs = outputs.strip()

        # print(f"\033[91m== Question: \033[0m\n{prompt}\n")
        # print(f"\033[91m== Response: \033[0m\n{outputs}\n")
        
        if chat_history is None:
            chat_history = []

        chat_history.append({"role":conv.roles[0], "content":user_prompt})
        chat_history.append({"role":conv.roles[1], "content":outputs})
        if return_history:
            return outputs, chat_history
        else:
            return outputs
        


    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


AutoConfig.register("videochat_flash_qwen", VideoChatFlashQwenConfig)
AutoModelForCausalLM.register(VideoChatFlashQwenConfig, VideoChatFlashQwenForCausalLM)
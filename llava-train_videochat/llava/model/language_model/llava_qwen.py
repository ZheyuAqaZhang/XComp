#    Copyright 2024 Hao Zhang
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
import copy
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM

# from .qwen.modeling_qwen import QWenLMHeadModel, QWenModel
# from .qwen.configuration_qwen import QWenConfig


class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing

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
                if 'EXTRA_PARAM_INNER_STRIDE_LIST' in os.environ:
                    inner_strids = eval(os.environ.get("EXTRA_PARAM_INNER_STRIDE_LIST", "[]"))
                    layer_list = [SelectTheNextHalf(stride=st) for st in inner_strids]
                else:
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

        self.post_init()
        
        if "EXTRA_PARAM_FREEZE_LAYER" in os.environ:
            for param in self.condenser.freeze_layers.parameters():
                param.requires_grad_(False)
            self.condenser.freeze_layers[0].mlp.gate_proj.weight.fill_(-1)

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
        first_conv_length=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # import pdb; pdb.set_trace()
        self.model.first_conv_length = first_conv_length

        # print("images[0].shape:", images[0].shape)
        if inputs_embeds is None:
            (input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels) = self.prepare_inputs_labels_for_multimodal(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)

        # print('!!!! VANILLA FORWARD !!!!')

        # import pdb; pdb.set_trace()
        # print("inputs_embeds.shape:", inputs_embeds.shape)
        # print("  Current GPU memory usage:", torch.cuda.memory_allocated() / 1024 / 1024 / 1024, "GB")
        # print("  Max GPU memory usage:", torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024, "GB")
        print(f"Shape {inputs_embeds.shape},  usage (cur/max): {torch.cuda.memory_allocated() / 1024 / 1024 / 1024:.2f} / {torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024:.2f} GB")

        # print("inputs_embeds.shape:", inputs_embeds.shape)
        if dpo_forward:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            return logits, labels

        else:
            res = super().forward(
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
                condenser=self.condenser,
            )
            return res

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (inputs, position_ids, attention_mask, _, inputs_embeds, _) = self.prepare_inputs_labels_for_multimodal(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes=image_sizes)
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs
    
    @torch.no_grad()
    def chat(self,
        video_path,
        tokenizer,
        user_prompt,
        chat_history=None,
        return_history=True,
        max_num_frames=512,
        media_dict=None,
        generation_config={}):

        frames, time_msg  = load_video(video_path, max_num_frames=max_num_frames, media_dict=media_dict)

        image_sizes = [frames[0].shape[:2]]

        frames = [self.get_vision_tower().image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self.model.dtype).cuda()]

        conv = conv_templates["qwen_2"].copy()

        if chat_history is None or len(chat_history) == 0:
            user_prompt = f'{DEFAULT_IMAGE_TOKEN}\n{time_msg.strip()} {user_prompt}'
        else:
            assert DEFAULT_IMAGE_TOKEN in chat_history[0]['content'], chat_history
            for msg in chat_history:
                conv.append_message(msg['role'], msg['content'])
        
        conv.append_message(conv.roles[0], user_prompt)
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        if tokenizer.pad_token_id is None:
            if "qwen" in tokenizer.name_or_path.lower():
                print("Setting pad token to bos token for qwen model.")
                tokenizer.pad_token_id = 151643

        attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = self.generate(
                inputs=input_ids,
                images=frames,
                attention_mask=attention_masks,
                modalities=["video"],
                image_sizes=image_sizes,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
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


AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)

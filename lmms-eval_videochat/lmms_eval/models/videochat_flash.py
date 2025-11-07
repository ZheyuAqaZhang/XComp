import logging
import warnings
from datetime import timedelta
from typing import List, Optional, Union, Tuple
import PIL

import torch
from tqdm import tqdm
from packaging import version


from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from transformers import AutoTokenizer, AutoModel

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model


# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
eval_logger = logging.getLogger("lmms-eval")

# Enable TF32 for CUDA
torch.backends.cuda.matmul.allow_tf32 = True


# Determine best attention implementation
if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


@register_model("videochat_flash")
class VideoChat_Flash(lmms):
    """
    VideoChat Flash
    """

    def __init__(
        self,
        pretrained: str = "xxx",
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        device_map: Optional[str] = "cuda:0",
        use_cache: Optional[bool] = True,
        max_num_frames: Optional[int] = 32,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now

        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        assert torch.cuda.device_count() > 0, torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"


        self.max_num_frames = max_num_frames

        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
        self._model = AutoModel.from_pretrained(pretrained, trust_remote_code=True).half().cuda()

        # modify here to use video-level compress
        self.model.config.mm_llm_compress = True
        self.model.config.llm_compress_type = "attention"
        self.model.config.llm_compress_layer_list = [4, 18]
        self.model.config.llm_image_token_ratio_list = [1 , 12/16, 4/16]
        # self.model.config.llm_compress_layer_list = []
        # self.model.config.llm_image_token_ratio_list = [1]
        # import math
        # cur_len = 16
        # n_layers = 28
        # for i in range(1, n_layers + 1):
        #     goal_len = 1 + int(15 * 0.5 * (1 + math.cos(i / n_layers * math.pi))+0.5)
        #     if cur_len != goal_len:
        #         self.model.config.llm_compress_layer_list.append(i)
        #         self.model.config.llm_image_token_ratio_list.append(goal_len / 16)
        #     cur_len = goal_len
        # print(f"llm_compress_layer_list: {self.model.config.llm_compress_layer_list}")
        # print(f"llm_image_token_ratio_list: {self.model.config.llm_image_token_ratio_list}")


        self._config = self._model.config
        self.model.eval()

        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache

        assert self.batch_size_per_gpu == 1

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._world_size = 1

        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length



    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError
  

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        metadata = requests[0].metadata 

        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        for chunk in chunks:
            batched_contexts, all_gen_kwargs, batched_doc_to_visual, batched_doc_id, batched_task, batched_split = zip(*chunk)

            task = batched_task[0]
            split = batched_split[0]
            batched_visuals = [batched_doc_to_visual[0](self.task_dict[task][split][ids]) for ids in batched_doc_id]  # [B, N]
            assert len(batched_visuals) == 1

            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]
            if "until" in gen_kwargs:
                gen_kwargs.pop("until")

            # preconfigure gen_kwargs with defaults
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "do_sample" not in gen_kwargs:
                gen_kwargs["do_sample"] = False
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1

            question_input = []
            text_outputs = []
            for visual, context in zip(batched_visuals, batched_contexts):
                if len(visual) > 1 or "image_aspect_ratio" not in self._config.__dict__:  # for multi image case, we treat per image aspect ratio as "pad" by default.
                    self._config.image_aspect_ratio = getattr(gen_kwargs, "image_aspect_ratio", "pad")
                    eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")

                if type(visual[0]) == PIL.Image.Image: # and "task_type" not in metadata and "sample_frames" not in metadata:  # For image task
                    raise NotImplementedError(f"I don't want image task now: {visual}, {task}, {metadata}")

                elif type(visual[0]) == str:  # For video task
                    if len(visual) > 1:
                        assert len(visual) == 2, visual
                        media_dict = visual[1]
                    else:
                        media_dict = {'video_read_type': 'decord'}

                    video_path = visual[0]
                    question_input.append(context)

                    try:
                        # import pdb; pdb.set_trace()
                        # import types
                        # from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, get_anyres_image_grid_shape, load_video
                        
                        # @torch.no_grad()
                        # def chat_patch(self,
                        #     video_path,
                        #     tokenizer,
                        #     user_prompt,
                        #     chat_history=None,
                        #     return_history=True,
                        #     max_num_frames=512,
                        #     media_dict=None,
                        #     generation_config={}):

                        #     frames, time_msg  = load_video(video_path, max_num_frames=max_num_frames, media_dict=media_dict)

                        #     image_sizes = [frames[0].shape[:2]]

                        #     frames = [self.get_vision_tower().image_processor.preprocess(frames, return_tensors="pt")["pixel_values"].to(self.model.dtype).cuda()]

                        #     conv = conv_templates["qwen_2"].copy()

                        #     if chat_history is None or len(chat_history) == 0:
                        #         user_prompt = f'{DEFAULT_IMAGE_TOKEN}\n{time_msg.strip()} {user_prompt}'
                        #     else:
                        #         assert DEFAULT_IMAGE_TOKEN in chat_history[0]['content'], chat_history
                        #         for msg in chat_history:
                        #             conv.append_message(msg['role'], msg['content'])
                            
                        #     conv.append_message(conv.roles[0], user_prompt)
                        #     conv.append_message(conv.roles[1], None)

                        #     prompt = conv.get_prompt()

                        #     input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

                        #     if tokenizer.pad_token_id is None:
                        #         if "qwen" in tokenizer.name_or_path.lower():
                        #             print("Setting pad token to bos token for qwen model.")
                        #             tokenizer.pad_token_id = 151643

                        #     attention_masks = input_ids.ne(tokenizer.pad_token_id).long().cuda()

                        #     stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                        #     keywords = [stop_str]
                        #     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
                            
                        #     with torch.inference_mode():
                        #         output_ids = self.generate(
                        #             inputs=input_ids,
                        #             images=frames,
                        #             attention_mask=attention_masks,
                        #             modalities=["video"],
                        #             image_sizes=image_sizes,
                        #             use_cache=True,
                        #             stopping_criteria=[stopping_criteria],
                        #             **generation_config
                        #         )

                        #     outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                        #     if outputs.endswith(stop_str):
                        #         outputs = outputs[: -len(stop_str)]

                        #     outputs = outputs.strip()

                        #     # print(f"\033[91m== Question: \033[0m\n{prompt}\n")
                        #     # print(f"\033[91m== Response: \033[0m\n{outputs}\n")
                            
                        #     if chat_history is None:
                        #         chat_history = []

                        #     chat_history.append({"role":conv.roles[0], "content":user_prompt})
                        #     chat_history.append({"role":conv.roles[1], "content":outputs})
                        #     if return_history:
                        #         return outputs, chat_history
                        #     else:
                        #         return outputs

                        # # 将 chat_patch 动态绑定为 self.model 的 chat 方法
                        # self.model.chat = types.MethodType(chat_patch, self.model)

                        response = self.model.chat(
                                video_path,
                                self.tokenizer,
                                context,
                                chat_history=None,
                                return_history=False,
                                max_num_frames=self.max_num_frames,
                                media_dict=media_dict,
                                generation_config={
                                    "max_new_tokens":gen_kwargs["max_new_tokens"],
                                    "temperature":gen_kwargs["temperature"],
                                    "do_sample":gen_kwargs["do_sample"],
                                    "top_p":gen_kwargs["top_p"],
                                    "num_beams":gen_kwargs["num_beams"]}
                                )
                        
                        text_outputs.append(response)
                    except Exception as e:
                        raise e

            text_outputs = [response.strip() for response in text_outputs]
            res.extend(text_outputs)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

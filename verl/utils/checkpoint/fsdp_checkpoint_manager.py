# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import warnings
from typing import Optional, Union

import torch
import torch.distributed
from accelerate import init_empty_weights
from torch.distributed.fsdp import FullStateDictConfig, ShardedOptimStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, PreTrainedTokenizer, ProcessorMixin

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local
from verl.utils.fsdp_utils import fsdp_version, get_fsdp_state_ctx

from .checkpoint_manager import BaseCheckpointManager


class FSDPCheckpointManager(BaseCheckpointManager):
    """
    Manage FSDP checkpointing in SPMD training.

    - Saves/loads per-rank sharded model & optimizer states
    - Persists full lr_scheduler and RNG state
    - Stores HF tokenizer/processor and model/config for unified restore

    Args:
        model (FSDP): Wrapped model instance.
        optimizer (Optimizer): Training optimizer.
        lr_scheduler (LRScheduler): Learning-rate scheduler.
        processing_class (PreTrainedTokenizer or ProcessorMixin, optional):
            Pre-/post-processing artifact handler.
        checkpoint_contents (list[str], optional):
            Components to include; 'model' and 'extra' are required. Include
            'optimizer' for exact training resume; omit it for final evaluation.
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
        **kwargs,
    ):
        if checkpoint_contents is None:
            checkpoint_contents = ["model", "optimizer", "extra"]
        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn("`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2)
            processing_class = kwargs.pop("tokenizer")
        if not {"model", "extra"}.issubset(checkpoint_contents):
            raise ValueError("FSDP checkpoints must include 'model' and 'extra'")
        unknown = set(checkpoint_contents) - {"model", "optimizer", "extra", "hf_model"}
        if unknown:
            raise ValueError(f"Unsupported checkpoint contents: {sorted(unknown)}")

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_contents=checkpoint_contents,
        )

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load an FSDP checkpoint for this rank.

        Downloads and loads:
          - model shards and optimizer shards when requested
          - extra state dict (scheduler + RNG)

        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        include_optimizer = "optimizer" in self.checkpoint_contents
        filenames = {
            "model": f"model_world_size_{self.world_size}_rank_{self.rank}.pt",
            "extra": f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt",
        }
        if include_optimizer:
            filenames["optimizer"] = f"optim_world_size_{self.world_size}_rank_{self.rank}.pt"

        states = {}
        downloaded_paths = []
        for component, filename in filenames.items():
            remote_path = os.path.join(local_path, filename)
            print(f"[rank-{self.rank}]: Loading {component} from {remote_path}")
            component_path = copy_to_local(remote_path)
            downloaded_paths.append(component_path)
            # Restore on CPU so saved device indices cannot concentrate all rank
            # shards on one GPU before FSDP redistributes the model state.
            states[component] = torch.load(component_path, weights_only=False, map_location="cpu")

        if del_local_after_load:
            try:
                for component_path in downloaded_paths:
                    if is_non_local(component_path):
                        os.remove(component_path)
            except Exception as e:
                print(f"[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored")

        model_state_dict = states["model"]
        extra_state_dict = states["extra"]
        lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]

        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=bool(is_cuda_available)) if include_optimizer else None
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
            self.model.load_state_dict(model_state_dict)
            if include_optimizer and self.optimizer is not None:
                self.optimizer.load_state_dict(states["optimizer"])
        # recover random state
        if "rng" in extra_state_dict:
            # 'rng' may not exist for backward compatibility
            self.load_rng_state(extra_state_dict["rng"])

        if self.lr_scheduler is not None and lr_scheduler_state_dict is not None:
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """
        Save an FSDP checkpoint for this rank.

        Writes:
          - model shards, plus optimizer shards when requested
          - extra state dict (scheduler + RNG)
          - HF tokenizer/processor and model/config on rank 0
          - optional full HF model under 'huggingface/' if requested

        Rotates old checkpoints, keeping at most `max_ckpt_to_keep`.

        Args:
            local_path: Target directory for checkpoint files.
            hdfs_path: Unused (for API compatibility).
            global_step: Current training step (used for bookkeeping).
            max_ckpt_to_keep: Number of recent checkpoints to retain.
        """
        if local_path is None:
            return

        # record the previous global step
        self.previous_global_step = global_step

        # remove previous local_path
        if max_ckpt_to_keep and isinstance(max_ckpt_to_keep, int) and max_ckpt_to_keep > 0 and len(self.previous_saved_paths) >= max_ckpt_to_keep:
            keep_start = len(self.previous_saved_paths) - max_ckpt_to_keep + 1
            self.remove_previous_save_local_path(self.previous_saved_paths[:keep_start])
            self.previous_saved_paths = self.previous_saved_paths[keep_start:]

        local_path = self.local_mkdir(local_path)
        torch.distributed.barrier()

        # Every rank saves its own model shard; optimizer state is optional.
        include_optimizer = "optimizer" in self.checkpoint_contents
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        optim_cfg = ShardedOptimStateDictConfig(offload_to_cpu=bool(is_cuda_available)) if include_optimizer else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, optim_cfg):
                model_state_dict = self.model.state_dict()
                optimizer_state_dict = self.optimizer.state_dict() if include_optimizer and self.optimizer is not None else None
                lr_scheduler_state_dict = self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None

                extra_state_dict = {
                    "lr_scheduler": lr_scheduler_state_dict,
                    "rng": self.get_rng_state(),
                }
                model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
                optim_path = os.path.join(local_path, f"optim_world_size_{self.world_size}_rank_{self.rank}.pt")
                extra_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")

                print(f"[rank-{self.rank}]: Saving model to {os.path.abspath(model_path)}")
                print(f"[rank-{self.rank}]: Saving extra_state to {os.path.abspath(extra_path)}")
                torch.save(model_state_dict, model_path)
                if include_optimizer:
                    print(f"[rank-{self.rank}]: Saving optim to {os.path.abspath(optim_path)}")
                    torch.save(optimizer_state_dict, optim_path)
                torch.save(extra_state_dict, extra_path)

        if self.rank == 0:
            if fsdp_version(self.model) == 1:
                unwrap_model = self.model._fsdp_wrapped_module
            else:
                unwrap_model = self.model

            model_config = unwrap_model.config
            if unwrap_model.can_generate() and hasattr(model_config, "name_or_path") and model_config.name_or_path:
                # Some model's name_or_path is empty if not initialized from pretrained,
                # in this cases, we don't save generation config.
                generation_config = GenerationConfig.from_pretrained(model_config.name_or_path)
                generation_config.save_pretrained(local_path)
            else:
                generation_config = None

            model_config.save_pretrained(local_path)
            self.processing_class.save_pretrained(local_path)

        # wait for everyone to dump to local
        torch.distributed.barrier()

        if "hf_model" in self.checkpoint_contents:
            hf_local_path = os.path.join(local_path, "huggingface")
            os.makedirs(hf_local_path, exist_ok=True)

            # Only rank 0 will save hf model and,
            # offload to cpu to save LLMs which may be too large to fit in one GPU
            state_dict_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with get_fsdp_state_ctx(self.model, StateDictType.FULL_STATE_DICT, state_dict_config, None):
                state_dict = self.model.state_dict()

            if self.rank == 0:
                if "ForTokenClassification" in model_config.architectures[0]:
                    from transformers import AutoModelForTokenClassification

                    auto_model_cls = AutoModelForTokenClassification
                elif "ForCausalLM" in model_config.architectures[0]:
                    from transformers import AutoModelForCausalLM

                    auto_model_cls = AutoModelForCausalLM
                elif "ForConditionalGeneration" in model_config.architectures[0]:
                    from transformers import AutoModelForVision2Seq

                    auto_model_cls = AutoModelForVision2Seq
                else:
                    raise NotImplementedError(f"Unknown architecture {model_config['architectures']}")

                with init_empty_weights():
                    save_model = auto_model_cls.from_config(model_config, torch_dtype=torch.bfloat16)
                save_model.to_empty(device="cpu")

                if save_model.can_generate():
                    if generation_config is not None:
                        save_model.generation_config = generation_config
                    else:
                        print(f"Warning: {self.__class__.__name__}.save_checkpoint: Generation config file not found in, using a generation config created from the model config when saving hf_model.")

                save_model.save_pretrained(hf_local_path, state_dict=state_dict)
                self.processing_class.save_pretrained(hf_local_path)
                del state_dict
                del save_model

            # wait for rank0 to dump hf_model to local
            torch.distributed.barrier()

        self.previous_saved_paths.append(local_path)

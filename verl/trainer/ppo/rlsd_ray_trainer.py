"""Shared teacher forward pass with privileged skill information."""

import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.rlsd_utils import SkillProvider
from verl.utils.model import compute_position_id_with_mask


def build_teacher_batch(
    batch: DataProto,
    skill_provider: SkillProvider,
    tokenizer,
    max_prompt_length: int,
    truncation: str = "error",
):
    """
    Build a teacher batch by prepending privileged skill info to each sample's prompt.

    The teacher sees (x, r) where r is the skill/privileged information.
    We prepend the skill text as a system message before the user prompt,
    then re-tokenize to get teacher input_ids/attention_mask/position_ids.
    The responses remain unchanged.

    Args:
        batch: The original student batch with prompts and responses.
        skill_provider: SkillProvider instance for loading skills.
        tokenizer: The tokenizer.
        max_prompt_length: Maximum prompt length.
        truncation: Truncation mode.

    Returns:
        teacher_batch: A DataProto with modified input_ids/attention_mask/position_ids
            but the same responses, suitable for computing teacher log probs.
    """
    bs = batch.batch["input_ids"].size(0)
    response_length = batch.batch["responses"].size(1)

    teacher_input_ids_list = []
    teacher_attention_mask_list = []
    teacher_position_ids_list = []

    for i in range(bs):
        # Decode the original prompt (student input minus response)
        original_input_ids = batch.batch["input_ids"][i]
        original_attention_mask = batch.batch["attention_mask"][i]
        prompt_length = original_input_ids.size(0) - response_length

        prompt_ids = original_input_ids[:prompt_length]
        prompt_mask = original_attention_mask[:prompt_length]

        # Find the first non-padding token in prompt
        valid_start = prompt_mask.nonzero(as_tuple=True)[0]
        if len(valid_start) > 0:
            valid_start = valid_start[0].item()
        else:
            valid_start = 0

        valid_prompt_ids = prompt_ids[valid_start:]
        prompt_text = tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)

        # Get privileged skill info based on gamefile, data_source, or prompt text
        gamefile = batch.non_tensor_batch.get("gamefile", None)
        data_source = batch.non_tensor_batch.get("data_source", None)
        if gamefile is not None:
            gf = gamefile[i]
            if gf is not None:
                gf = gf if isinstance(gf, str) else str(gf)
                skill_text = skill_provider.get_privileged_info(gf)
            elif data_source is not None:
                ds = data_source[i] if isinstance(data_source[i], str) else str(data_source[i])
                skill_text = skill_provider.get_privileged_info_from_data_source(ds, prompt_text)
            else:
                skill_text = skill_provider.get_privileged_info_from_prompt(prompt_text)
        elif data_source is not None:
            ds = data_source[i] if isinstance(data_source[i], str) else str(data_source[i])
            skill_text = skill_provider.get_privileged_info_from_data_source(ds, prompt_text)
        else:
            skill_text = skill_provider.get_privileged_info_from_prompt(prompt_text)

        # Construct teacher prompt: prepend skill as a system message
        skill_prefix = f"[Privileged Skill Information]\n{skill_text}\n\n"
        teacher_prompt_text = skill_prefix + prompt_text

        # Tokenize the teacher prompt
        teacher_prompt_ids = tokenizer.encode(teacher_prompt_text, add_special_tokens=False)

        # Truncate if needed (left truncation to keep the end of prompt)
        if len(teacher_prompt_ids) > max_prompt_length:
            teacher_prompt_ids = teacher_prompt_ids[-max_prompt_length:]

        teacher_prompt_ids = torch.tensor(teacher_prompt_ids, dtype=torch.long)
        actual_prompt_len = len(teacher_prompt_ids)

        # Pad to max_prompt_length (left padding)
        pad_length = max_prompt_length - actual_prompt_len
        if pad_length > 0:
            pad_ids = torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)
            teacher_prompt_ids = torch.cat([pad_ids, teacher_prompt_ids])
            t_prompt_mask = torch.cat([
                torch.zeros(pad_length, dtype=torch.long),
                torch.ones(actual_prompt_len, dtype=torch.long),
            ])
        else:
            t_prompt_mask = torch.ones(actual_prompt_len, dtype=torch.long)

        # Combine with response
        response_ids = batch.batch["responses"][i]
        response_mask = original_attention_mask[-response_length:]

        teacher_full_ids = torch.cat([teacher_prompt_ids, response_ids])
        teacher_full_mask = torch.cat([t_prompt_mask, response_mask])
        teacher_position_ids = compute_position_id_with_mask(teacher_full_mask.unsqueeze(0))[0]

        teacher_input_ids_list.append(teacher_full_ids)
        teacher_attention_mask_list.append(teacher_full_mask)
        teacher_position_ids_list.append(teacher_position_ids)

    teacher_input_ids = torch.stack(teacher_input_ids_list)
    teacher_attention_mask = torch.stack(teacher_attention_mask_list)
    teacher_position_ids = torch.stack(teacher_position_ids_list)

    teacher_batch = DataProto.from_dict(
        tensors={
            "input_ids": teacher_input_ids,
            "attention_mask": teacher_attention_mask,
            "position_ids": teacher_position_ids,
            "responses": batch.batch["responses"],
        },
    )

    return teacher_batch


class RLSDRayTrainer(RayPPOTrainer):
    """
    Shared skill-conditioned teacher support for the training loop.
    """

    def __init__(self, *args, skill_provider: SkillProvider = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.skill_provider = skill_provider

    def _compute_teacher_log_probs(self, batch: DataProto) -> torch.Tensor:
        """
        Compute teacher log probs by running forward pass with privileged skill info
        prepended to the prompt. Uses the same model π_θ but conditioned on (x, r).
        """
        teacher_batch = build_teacher_batch(
            batch=batch,
            skill_provider=self.skill_provider,
            tokenizer=self.tokenizer,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation=self.config.data.get("truncation", "left"),
        )

        # Use the same actor to compute teacher log probs
        teacher_output = self.actor_rollout_wg.compute_log_prob(teacher_batch)
        teacher_log_probs = teacher_output.batch["old_log_probs"]

        return teacher_log_probs

"""Run the fixed CORD recipes with practical resource controls."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading
import uuid

from .recipes import METHODS, MODEL, TASKS

ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("train", "evaluate"))
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--model", default=MODEL, help="Model identifier or local model directory.")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--tensor-parallel", type=int, default=2)
    p.add_argument("--micro-batch-size", type=int, default=4, help="Memory control for actor forward/backward.")
    p.add_argument("--logprob-micro-batch-size", type=int, default=4, help="Memory control for scoring.")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--checkpoint", type=Path, help="Evaluation checkpoint directory global_step_N.")
    return p


def _override(key, value):
    if isinstance(value, bool):
        value = str(value).lower()
    elif isinstance(value, str):
        value = json.dumps(value)
    return f"{key}={value}"


def plan(args, run_dir):
    cfg = TASKS[args.task]
    training = args.mode == "train"
    steps = 150
    train_size = 16
    val_size = 128 if training else cfg["final_size"]
    manifest = ROOT / "manifests" / cfg["validation" if training else "test"]
    if not manifest.is_file():
        raise ValueError(f"Missing bundled manifest: {manifest.name}")
    entries = json.loads(manifest.read_text())
    if len(entries) < val_size:
        raise ValueError(f"Manifest has {len(entries)} entries; {val_size} required.")
    if args.task == "webshop":
        if not all(isinstance(i, int) and 0 <= i < 500 for i in entries):
            raise ValueError("WebShop manifest must contain test goal IDs 0 through 499.")
    elif any(Path(i).is_absolute() or ".." in Path(i).parts for i in entries):
        raise ValueError("ALFWorld manifest must contain relative dataset paths.")
    checkpoint = args.checkpoint or args.output_dir / "checkpoints" / f"global_step_{steps}"
    values = {
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "+algorithm.cord.recipe": args.method,
        "+algorithm.cord.skills_dir": str(ROOT / "skills" / args.task),
        "data.train_files": str(run_dir / "data/train.parquet"),
        "data.val_files": str(run_dir / "data/test.parquet"),
        "data.train_batch_size": train_size,
        "data.val_batch_size": val_size,
        "data.max_prompt_length": cfg["prompt_tokens"],
        "data.max_response_length": 512,
        "data.filter_overlong_prompts": True,
        "data.truncation": "error",
        "data.return_raw_chat": True,
        "+data.seed": 0,
        "actor_rollout_ref.model.path": args.model,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.optim.lr": 1e-6,
        "actor_rollout_ref.actor.checkpoint.contents": ["model", "extra"],
        "actor_rollout_ref.actor.ppo_mini_batch_size": cfg["mini_batch"],
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": args.micro_batch_size,
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": 0.01,
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.use_invalid_action_penalty": True,
        "actor_rollout_ref.actor.invalid_action_penalty_coef": 0.1,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": args.logprob_micro_batch_size,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": args.tensor_parallel,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.6,
        "actor_rollout_ref.rollout.enable_chunked_prefill": False,
        "actor_rollout_ref.rollout.enforce_eager": False,
        "actor_rollout_ref.rollout.free_cache_engine": False,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0 if training else 0.4,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": not training,
        "actor_rollout_ref.rollout.val_kwargs.top_p": 1.0,
        "actor_rollout_ref.rollout.val_kwargs.top_k": -1,
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": args.logprob_micro_batch_size,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "env.env_name": cfg["env"],
        "env.seed": 0,
        "env.max_steps": cfg["max_steps"],
        "env.history_length": 2,
        "env.rollout.n": 8,
        "env.resources_per_worker.num_cpus": 0.1,
        f"+env.{args.task}.fixed_val": True,
        f"+env.{args.task}." + ("eval_manifest" if args.task == "alfworld" else "val_goal_manifest"): str(manifest),
        "trainer.critic_warmup": 0,
        "trainer.logger": "[console]",
        "trainer.project_name": "cord",
        "trainer.experiment_name": f"{args.task}-{args.method}",
        "trainer.n_gpus_per_node": args.gpus,
        "trainer.nnodes": 1,
        "trainer.ray_wait_register_center_timeout": 600,
        "trainer.save_freq": steps if training else -1,
        "trainer.resume_mode": "disable" if training else "resume_path",
        "trainer.default_local_dir": str(args.output_dir / "checkpoints"),
        "trainer.val_only": not training,
        "trainer.val_repeat_n": 1 if training else 4,
        "trainer.test_freq": 5 if training else -1,
        "trainer.total_epochs": steps,
        "trainer.val_before_train": True,
        "+trainer.evaluation_output_dir": str(run_dir / "evaluation"),
        "hydra.run.dir": str(run_dir / "hydra"),
        "hydra.job.chdir": False,
    }
    if args.task == "webshop":
        values["env.webshop.use_small"] = True
        values["env.webshop.human_goals"] = False
    if not training:
        values["trainer.resume_from_path"] = str(checkpoint)
    cmd = [sys.executable, "-m", "verl.trainer.main_cord"]
    cmd += [_override(k, v) for k, v in values.items() if k != "trainer.logger"]
    cmd.append("trainer.logger=[console]")
    return cmd, entries[:val_size], train_size, val_size, checkpoint, values


def storage_budget_gib(args):
    """Bound the released 3B recipe; use a larger budget for other models."""
    model = Path(args.model).expanduser()
    local = model.is_dir()
    small_model = args.model == MODEL
    if local:
        for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index = model / name
            if index.is_file():
                size = json.loads(index.read_text()).get("metadata", {}).get("total_size", 0)
                small_model = 0 < size <= 7 * 1024 ** 3
                break
        else:
            weights = model / "model.safetensors"
            if weights.is_file():
                small_model = 0 < weights.stat().st_size <= 7 * 1024 ** 3
    if not small_model:
        return 100
    if args.mode == "evaluate":
        return 4 if local else 12
    return 32


def validate_assets(args, entries):
    if args.task == "alfworld":
        data = Path(os.environ.get("ALFWORLD_DATA", str(ROOT / "data/alfworld"))).expanduser()
        required = [data / p for p in entries]
        required += [data / "logic/alfred.pddl", data / "logic/alfred.twl2", data / "json_2.1.1/train"]
        os.environ["ALFWORLD_DATA"] = str(data.resolve())
    else:
        data = Path(os.environ.get("WEBSHOP_DATA", str(ROOT / "data/webshop"))).expanduser()
        index = Path(os.environ.get("WEBSHOP_INDEX", str(ROOT / "data/webshop/indexes"))).expanduser()
        required = [data / name for name in ("items_shuffle_1000.json", "items_ins_v2_1000.json", "items_human_ins.json")]
        required.append(index)
        os.environ["WEBSHOP_DATA"] = str(data.resolve())
        os.environ["WEBSHOP_INDEX"] = str(index.resolve())
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError("Benchmark assets are missing; follow README setup instructions: " + ", ".join(missing[:3]))


def runtime_env(output):
    env = os.environ.copy()
    runtime = output / ".runtime"
    temporary = ROOT / "t" / hashlib.sha256(str(output).encode()).hexdigest()[:8]
    paths = {
        "HF_HOME": runtime / "huggingface",
        "HF_HUB_CACHE": runtime / "huggingface/hub",
        "HF_DATASETS_CACHE": runtime / "huggingface/datasets",
        "TRANSFORMERS_CACHE": runtime / "huggingface/hub",
        "XDG_CACHE_HOME": runtime / "cache",
        "TORCH_HOME": runtime / "torch",
        "TORCHINDUCTOR_CACHE_DIR": runtime / "torchinductor",
        "TRITON_CACHE_DIR": runtime / "triton",
        "VLLM_CACHE_ROOT": runtime / "vllm",
        "TMPDIR": temporary,
        "TEMP": temporary,
        "TMP": temporary,
        "RAY_TMPDIR": ROOT / "r",
    }
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HYDRA_FULL_ERROR"] = "1"
    env["RAY_USAGE_STATS_ENABLED"] = "0"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["VLLM_DO_NOT_TRACK"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def execute(cmd, env, output, run_dir, budget):
    from .storage import RESERVE, tree_bytes
    import shutil

    log_limit = 32 * 1024 * 1024
    stop = threading.Event()
    violation = []
    with (run_dir / "run.log").open("w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                start_new_session=True)

        def monitor():
            while not stop.wait(10):
                if shutil.disk_usage(output).free < RESERVE:
                    violation.append("Free space dropped below the 64 GiB reserve.")
                elif sum(tree_bytes(Path(p)) for p in budget["tracked_roots"]) > budget["maximum_retained_bytes"]:
                    violation.append("Run output exceeded its registered storage budget.")
                if violation:
                    os.killpg(proc.pid, signal.SIGTERM)
                    return

        watcher = threading.Thread(target=monitor, daemon=True)
        watcher.start()
        written = 0
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                if written < log_limit:
                    text = line[:log_limit - written]
                    log.write(text)
                    written += len(text.encode("utf-8"))
            result = proc.wait()
        except KeyboardInterrupt:
            os.killpg(proc.pid, signal.SIGTERM)
            result = proc.wait()
        finally:
            stop.set()
            watcher.join(timeout=2)
    if violation:
        (run_dir / "storage-stop.json").write_text(json.dumps(violation) + "\n")
        raise RuntimeError(violation[0])
    if result:
        raise RuntimeError(f"Run exited with status {result}; see {run_dir / 'run.log'}.")


def main():
    p = parser()
    args = p.parse_args()
    if min(args.gpus, args.tensor_parallel, args.micro_batch_size, args.logprob_micro_batch_size) < 1:
        p.error("GPU, tensor-parallel and micro-batch counts must be positive.")
    if args.gpus % args.tensor_parallel:
        p.error("--gpus must be divisible by --tensor-parallel.")
    minibatch = TASKS[args.task]["mini_batch"]
    if minibatch % (args.gpus * args.micro_batch_size):
        p.error("PPO mini-batch must divide evenly across GPUs and actor micro-batches.")
    if args.checkpoint and args.mode != "evaluate":
        p.error("--checkpoint is for evaluation.")
    args.output_dir = (args.output_dir or ROOT / "outputs" /
                       f"{args.task}-{args.method}").expanduser().resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve()
    suffix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = args.output_dir / "runs" / f"{args.mode}-{suffix}"
    cmd, entries, train_size, val_size, checkpoint, values = plan(args, run_dir)
    if args.mode == "train":
        ckpt_root = args.output_dir / "checkpoints"
        if ckpt_root.exists() and any(ckpt_root.iterdir()):
            p.error("Checkpoint output already exists; choose a new --output-dir. Existing files are retained.")
    elif not (checkpoint / "actor").is_dir():
        p.error(f"Missing checkpoint actor directory: {checkpoint / 'actor'}")
    else:
        expected = [checkpoint / "actor" / f"model_world_size_{args.gpus}_rank_{rank}.pt" for rank in range(args.gpus)]
        if any(not path.is_file() for path in expected):
            p.error("Checkpoint model shards are incomplete or were saved with a different GPU count.")
    validate_assets(args, entries)
    run_dir.mkdir(parents=True, exist_ok=False)
    from .storage import preflight
    budget = preflight(args.output_dir, run_dir, args.mode == "train",
                       (ROOT / "outputs", ROOT / "r", ROOT / "t"), additional_gib=storage_budget_gib(args))
    (run_dir / "config.json").write_text(json.dumps(values, indent=2) + "\n")
    (run_dir / "command.txt").write_text(shlex.join(cmd) + "\n")
    import importlib.metadata
    versions = {}
    for name in ("torch", "vllm", "ray", "transformers", "tensordict", "flash-attn", "alfworld"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    sources = {}
    for folder in ("cord", "verl", "agent_system", "skills", "manifests"):
        for path in (ROOT / folder).rglob("*"):
            if path.is_file() and not path.is_symlink() and path.stat().st_size <= 2 * 1024 * 1024 and path.suffix in (".py", ".yaml", ".json", ".md"):
                sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (run_dir / "provenance.json").write_text(json.dumps({"python": sys.version, "packages": versions,
                  "source_sha256": sources}, indent=2) + "\n")
    from .prepare import prepare
    prepare(run_dir / "data", train_size, val_size)
    env = runtime_env(args.output_dir)
    execute(cmd, env, args.output_dir, run_dir, budget)
    print(f"Completed. Run records: {run_dir}")


if __name__ == "__main__":
    main()

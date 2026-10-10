"""Fixed experiment recipes and task settings."""
MODEL = "Qwen/Qwen2.5-3B-Instruct"
METHODS = ("skillsd", "sdar")
TASKS = {
    "alfworld": {"env": "alfworld/AlfredTWEnv", "prompt_tokens": 2048,
                 "mini_batch": 256, "max_steps": 50, "final_size": 128,
                 "validation": "alfworld_validation.json", "test": "alfworld_validation.json"},
    "webshop": {"env": "Webshop", "prompt_tokens": 4096,
                "mini_batch": 64, "max_steps": 15, "final_size": 500,
                "validation": "webshop_validation.json", "test": "webshop_test.json"},
}

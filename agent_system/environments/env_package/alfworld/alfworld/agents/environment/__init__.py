def get_environment(env_type):
    if env_type != "AlfredTWEnv":
        raise ValueError("This release uses the ALFWorld text environment")
    from .alfred_tw_env import AlfredTWEnv
    return AlfredTWEnv

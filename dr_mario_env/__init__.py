from .simulator import SimulatorClient

__all__ = ["DrMarioEnv", "DrMarioCnnPolicy", "SimulatorClient"]


def __getattr__(name: str):
    if name == "DrMarioEnv":
        from .env import DrMarioEnv

        return DrMarioEnv
    if name == "DrMarioCnnPolicy":
        from .policy import DrMarioCnnPolicy

        return DrMarioCnnPolicy
    raise AttributeError(name)

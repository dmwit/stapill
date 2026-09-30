from .simulator import SimulatorClient

__all__ = ["DrMarioEnv", "SimulatorClient"]


def __getattr__(name: str):
    if name == "DrMarioEnv":
        from .env import DrMarioEnv

        return DrMarioEnv
    raise AttributeError(name)

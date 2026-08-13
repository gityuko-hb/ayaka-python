import torch

SUPPORTED_DEVICES = [
    "cpu",
    "cuda"
]


class DeviceConfig:
    device: torch.device | None

    def __init__(self, device: str = "cuda") -> None:
        if device in SUPPORTED_DEVICES:
            self.device_type = device
        else:
            raise RuntimeError(f"DeviceConfig is not supported device type: {device}")

        self.device = torch.device(self.device_type)

class ModelConfig:
    def __init__(
        self,
        model_path: str,
        dtype: str = "auto"
    ):
        self.model_path = model_path
        self.dtype = dtype
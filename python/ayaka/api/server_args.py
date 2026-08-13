import argparse
import dataclasses


@dataclasses.dataclass
class ServerArguments:
    model_path: str
    tokenizer_path: str | None = None
    random_seed: int = 42
    log_level: str = "info"
    host: str = "127.0.0.1"
    port: int = 8000

    def __post_init__(self):
        if self.tokenizer_path is None:
            self.tokenizer_path = self.model_path

    def get_server_url(self):
        return f"http://{self.host}:{self.port}"

    @classmethod
    def from_cli_args(
        cls,
        args: argparse.Namespace
    ):
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--model-path",
            type=str,
            help="The path of the model weights.",
            required=True,
        )
        parser.add_argument(
            "--tokenizer-path",
            type=str,
            default=ServerArguments.tokenizer_path,
            help="The path of the tokenizer.",
        )
        parser.add_argument(
            "--random-seed",
            type=int,
            default=ServerArguments.random_seed,
            help="Random seed.",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArguments.log_level,
            help="Log level",
        )
        parser.add_argument(
            "--host", 
            type=str, 
            default=ServerArguments.host
        )
        parser.add_argument(
            "--port", 
            type=int, 
            default=ServerArguments.port
        )
        
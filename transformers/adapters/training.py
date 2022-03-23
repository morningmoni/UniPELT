from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AdapterArguments:
    """
    The subset of arguments related to model training.
    """

    train_adapter: bool = field(default=False, metadata={"help": "Train an model instead of the full model."})
    load_adapter: Optional[str] = field(
        default="", metadata={"help": "Pre-trained model module to be loaded from Hub."}
    )
    adapter_config: Optional[str] = field(
        default="pfeiffer", metadata={"help": "Adapter configuration. Either an identifier or a path to a file."}
    )
    adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the model configuration."}
    )
    adapter_reduction_factor: Optional[int] = field(
        default=None, metadata={"help": "Override the reduction factor of the model configuration."}
    )
    language: Optional[str] = field(default=None, metadata={"help": "The training language, e.g. 'en' for English."})


@dataclass
class MultiLingAdapterArguments(AdapterArguments):
    """
    Arguemnts related to model training, extended by arguments for multilingual setups.
    """

    load_lang_adapter: Optional[str] = field(
        default=None, metadata={"help": "Pre-trained language model module to be loaded from Hub."}
    )
    lang_adapter_config: Optional[str] = field(
        default=None, metadata={"help": "Language model configuration. Either an identifier or a path to a file."}
    )
    lang_adapter_non_linearity: Optional[str] = field(
        default=None, metadata={"help": "Override the non-linearity of the language model configuration."}
    )
    lang_adapter_reduction_factor: Optional[int] = field(
        default=None, metadata={"help": "Override the reduction factor of the language model configuration."}
    )

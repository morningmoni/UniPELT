import itertools
from collections.abc import Sequence
from typing import List, Set, Union


class AdapterCompositionBlock(Sequence):
    def __init__(self, *children):
        self.children = [parse_composition(b, None) for b in children]

    def __getitem__(self, key):
        return self.children[key]

    def __len__(self):
        return len(self.children)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, type(self)):
            return all([c1 == c2 for c1, c2 in zip(self.children, o.children)])
        else:
            return False

    def __repr__(self):
        child_repr = ", ".join(map(str, self.children))
        return f"{self.__class__.__name__}[{child_repr}]"

    def first(self):
        if not isinstance(self.children[0], AdapterCompositionBlock):
            return self.children[0]
        else:
            return self.children[0].first()

    def last(self):
        if not isinstance(self.children[-1], AdapterCompositionBlock):
            return self.children[-1]
        else:
            return self.children[-1].last()

    @property
    def parallel_channels(self):
        return max([b.parallel_channels if isinstance(b, AdapterCompositionBlock) else 1 for b in self.children])

    def flatten(self) -> Set[str]:
        return set(itertools.chain(*[[b] if isinstance(b, str) else b.flatten() for b in self.children]))


class Parallel(AdapterCompositionBlock):
    def __init__(self, *parallel_adapters: List[str]):
        """
        Can be used to perform inference for multiple tasks (i.e., adapters) in parallel (for the same input).

        See AdapterDrop https://arxiv.org/abs/2010.11918
        """
        super().__init__(*parallel_adapters)

    @property
    def parallel_channels(self):
        return len(self.children)


class Stack(AdapterCompositionBlock):
    def __init__(self, *stack_layers: List[Union[AdapterCompositionBlock, str]]):
        super().__init__(*stack_layers)


class Fuse(AdapterCompositionBlock):
    def __init__(self, *fuse_stacks: List[Union[AdapterCompositionBlock, str]]):
        super().__init__(*fuse_stacks)

    # TODO-V2 pull this up to all block classes?
    @property
    def name(self):
        return ",".join([c if isinstance(c, str) else c.last() for c in self.children])


class Split(AdapterCompositionBlock):
    def __init__(self, left: str, right: str, split_index: int):
        super().__init__(left, right)
        assert split_index > 0
        self.left = left
        self.right = right
        self.split_index = split_index


# Mapping each composition block type to the allowed nested types
ALLOWED_NESTINGS = {
    Stack: [str, Fuse, Split, Parallel],
    Fuse: [str, Stack],
    Split: [str, Split, Stack],
    Parallel: [str, Stack],
}

# Some composition blocks might not be supported by all models.
# Add a whitelist of models for those here.
SUPPORTED_MODELS = {
    Parallel: ["bert", "roberta", "distilbert", "bart", "mbart", "gpt2"],
}


def validate_composition(adapter_composition: AdapterCompositionBlock, level=0, model_type=None):
    if level > 1 and not (isinstance(adapter_composition, Stack) or isinstance(adapter_composition, str)):
        raise ValueError(f"Adapter setup is too deep. Cannot have {adapter_composition} at level {level}.")
    if isinstance(adapter_composition, AdapterCompositionBlock):
        block_type = type(adapter_composition)
        if model_type and block_type in SUPPORTED_MODELS:
            if model_type not in SUPPORTED_MODELS[block_type]:
                raise ValueError(
                    f"Models of type {model_type} don't support model composition using {block_type.__name__}."
                )
        for child in adapter_composition:
            if not type(child) in ALLOWED_NESTINGS[type(adapter_composition)]:
                raise ValueError(f"Adapter setup is invalid. Cannot nest {child} in {adapter_composition}")
            # recursively validate children
            validate_composition(child, level=level + 1)


def parse_composition(adapter_composition, level=0, model_type=None) -> AdapterCompositionBlock:
    """
    Parses and validates a setup of adapters.

    Args:
        adapter_composition: The model setup to be parsed.
        level (int, optional): If set to none, disables validation. Defaults to 0.
    """
    if not adapter_composition:
        return None
    elif isinstance(adapter_composition, AdapterCompositionBlock):
        if level is not None:
            validate_composition(adapter_composition, level=level, model_type=model_type)
        return adapter_composition
    elif isinstance(adapter_composition, str):
        if level == 0:
            return Stack(adapter_composition)
        else:
            return adapter_composition
    elif isinstance(adapter_composition, Sequence):
        # for backwards compatibility
        if level == 1:
            block_class = Fuse
        else:
            block_class = Stack
        level = level + 1 if level is not None else None
        return block_class(*[parse_composition(b, level) for b in adapter_composition])
    else:
        raise TypeError(adapter_composition)

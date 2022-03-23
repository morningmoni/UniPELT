# UniPELT
This repo provides the code for paper ["UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning"](https://arxiv.org/abs/2110.07577), ACL 2022.

We support multiple Parameter-Efficient Language model Tuning (PELT) methods, including Prefix-tuning, Adapter, LoRA, BitFit, and any combination of them on BERT.



## How to run
Use `run_glue.py` as the entry file.

To use Prefix-tuning, set `--add_enc_prefix True`

To use Adapter, set `--train_adapter`

To use LoRA, set `--add_lora True`

To use BitFit, set `--tune_bias True`

The codebase is based on [transformers (adapter-transformers)](https://github.com/Adapter-Hub/adapter-transformers/). See [here](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L86) for more details of the training arguments.
Please also refer to the following repos: [Prefix-tuning](https://github.com/XiangLi1999/PrefixTuning), [LoRA](https://github.com/microsoft/LoRA).

## Reference
If you use the code for your work, please consider citing our paper.
```
@inproceedings{mao-etal-2022-unipelt,
  title={UniPELT: A Unified Framework for Parameter-Efficient Language Model Tuning},
  author={Mao, Yuning and Mathias, Lambert and Hou, Rui and Almahairi, Amjad and Ma, Hao and Han, Jiawei and Yih, Wen-tau and Khabsa, Madian},
  journal={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
  year={2022}
}
```

## License
The majority of UniPELT is licensed under CC-BY-NC, however portions of the project are available under separate license terms: transformers (adapter-transformers) is licensed under the Apache 2.0 license and LoRA is licensed under the MIT License.



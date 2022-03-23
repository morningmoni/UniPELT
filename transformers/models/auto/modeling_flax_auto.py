# coding=utf-8
# Copyright 2018 The Google Flax Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Auto Model class. """


from collections import OrderedDict

from ...utils import logging
from ..bert.modeling_flax_bert import (
    FlaxBertForMaskedLM,
    FlaxBertForMultipleChoice,
    FlaxBertForNextSentencePrediction,
    FlaxBertForPreTraining,
    FlaxBertForQuestionAnswering,
    FlaxBertForSequenceClassification,
    FlaxBertForTokenClassification,
    FlaxBertModel,
)
from ..roberta.modeling_flax_roberta import FlaxRobertaModel
from .auto_factory import auto_class_factory
from .configuration_auto import BertConfig, RobertaConfig


logger = logging.get_logger(__name__)


FLAX_MODEL_MAPPING = OrderedDict(
    [
        # Base model mapping
        (RobertaConfig, FlaxRobertaModel),
        (BertConfig, FlaxBertModel),
    ]
)

FLAX_MODEL_FOR_PRETRAINING_MAPPING = OrderedDict(
    [
        # Model for pre-training mapping
        (BertConfig, FlaxBertForPreTraining),
    ]
)

FLAX_MODEL_FOR_MASKED_LM_MAPPING = OrderedDict(
    [
        # Model for Masked LM mapping
        (BertConfig, FlaxBertForMaskedLM),
    ]
)

FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Sequence Classification mapping
        (BertConfig, FlaxBertForSequenceClassification),
    ]
)

FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING = OrderedDict(
    [
        # Model for Question Answering mapping
        (BertConfig, FlaxBertForQuestionAnswering),
    ]
)

FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = OrderedDict(
    [
        # Model for Token Classification mapping
        (BertConfig, FlaxBertForTokenClassification),
    ]
)

FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING = OrderedDict(
    [
        # Model for Multiple Choice mapping
        (BertConfig, FlaxBertForMultipleChoice),
    ]
)

FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = OrderedDict(
    [
        (BertConfig, FlaxBertForNextSentencePrediction),
    ]
)

FlaxAutoModel = auto_class_factory("FlaxAutoModel", FLAX_MODEL_MAPPING)

FlaxAutoModelForPreTraining = auto_class_factory(
    "FlaxAutoModelForPreTraining", FLAX_MODEL_FOR_PRETRAINING_MAPPING, head_doc="pretraining"
)

FlaxAutoModelForMaskedLM = auto_class_factory(
    "FlaxAutoModelForMaskedLM", FLAX_MODEL_FOR_MASKED_LM_MAPPING, head_doc="masked language modeling"
)

FlaxAutoModelForSequenceClassification = auto_class_factory(
    "AFlaxutoModelForSequenceClassification",
    FLAX_MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    head_doc="sequence classification",
)

FlaxAutoModelForQuestionAnswering = auto_class_factory(
    "FlaxAutoModelForQuestionAnswering", FLAX_MODEL_FOR_QUESTION_ANSWERING_MAPPING, head_doc="question answering"
)

FlaxAutoModelForTokenClassification = auto_class_factory(
    "FlaxAutoModelForTokenClassification", FLAX_MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
)

FlaxAutoModelForMultipleChoice = auto_class_factory(
    "AutoModelForMultipleChoice", FLAX_MODEL_FOR_MULTIPLE_CHOICE_MAPPING, head_doc="multiple choice"
)

FlaxAutoModelForNextSentencePrediction = auto_class_factory(
    "FlaxAutoModelForNextSentencePrediction",
    FLAX_MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    head_doc="next sentence prediction",
)

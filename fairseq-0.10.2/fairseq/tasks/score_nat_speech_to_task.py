import logging
import os.path as op
from argparse import Namespace

import torch

from fairseq.tasks.nat_speech_to_task import NATSpeechToTextTask

from fairseq.data import Dictionary, encoders


from fairseq.tasks.translation import load_langpair_dataset
from fairseq.tasks import FairseqTask, register_task
from fairseq.utils import new_arange


logger = logging.getLogger(__name__)

@register_task("score_nat_speech_to_text")
class ScoreNATSpeechToTextTask(NATSpeechToTextTask):
    def __init__(self, args, tgt_dict) -> None:
        super().__init__(args, tgt_dict)
    
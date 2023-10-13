from email.policy import default
import logging
import os.path as op
from argparse import Namespace

import torch

from fairseq.tasks.speech_to_text import SpeechToTextTask

from fairseq.data import Dictionary, encoders
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
)

from fairseq.tasks.translation import load_langpair_dataset
from fairseq.tasks import FairseqTask, register_task
from fairseq.utils import new_arange


logger = logging.getLogger(__name__)

@register_task("nat_speech_to_text")
class NATSpeechToTextTask(SpeechToTextTask):
    def __init__(self, args,tgt_dict):
        super().__init__(args,tgt_dict)
    
    @staticmethod
    def add_args(parser):
        SpeechToTextTask.add_args(parser)
        parser.add_argument(
            '--noise',
            default='random_delete',
            choices=['random_delete', 'random_mask', 'no_noise', 'full_mask'])
        parser.add_argument(
            '--retain_iter_history',
            default=False,
            action='store_true',
        )
        parser.add_argument(
            '--iter_decode_max_iter',
            type=int,
            default=1,
        )
        parser.add_argument(
            '--assessment',
            default=False,
            action='store_true',
        )
        parser.add_argument(
            '--score',
            default=False,
            action='store_true',
        )
        parser.add_argument(
            '--no_bpe',
            default=False,
            action='store_true',
        )
        parser.add_argument(
            '--mask_length',
            type=int,
            default=1,
        )
        # parser.add_argument(
        #     '--phoneme',
        #     default=False,
        #     action='store_true',
        # )

        
    def inject_noise(self, target_tokens):
        def _random_delete(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()

            max_len = target_tokens.size(1)
            target_mask = target_tokens.eq(pad)
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(
                target_tokens.eq(bos) | target_tokens.eq(eos), 0.0
            )
            target_score.masked_fill_(target_mask, 1)
            target_score, target_rank = target_score.sort(1)
            target_length = target_mask.size(1) - target_mask.float().sum(
                1, keepdim=True
            )

            # do not delete <bos> and <eos> (we assign 0 score for them)
            target_cutoff = (
                2
                + (
                    (target_length - 2)
                    * target_score.new_zeros(target_score.size(0), 1).uniform_()
                ).long()
            )
            target_cutoff = target_score.sort(1)[1] >= target_cutoff

            prev_target_tokens = (
                target_tokens.gather(1, target_rank)
                .masked_fill_(target_cutoff, pad)
                .gather(1, target_rank.masked_fill_(target_cutoff, max_len).sort(1)[1])
            )
            prev_target_tokens = prev_target_tokens[
                :, : prev_target_tokens.ne(pad).sum(1).max()
            ]

            return prev_target_tokens

        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * target_length.clone().uniform_()
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), unk
            )
            return prev_target_tokens

        def _full_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()

            target_mask = (
                target_tokens.eq(bos) | target_tokens.eq(eos) | target_tokens.eq(pad)
            )
            return target_tokens.masked_fill(~target_mask, unk)

        if self.args.noise == "random_delete":
            return _random_delete(target_tokens)
        elif self.args.noise == "random_mask":
            return _random_mask(target_tokens)
        elif self.args.noise == "full_mask":
            return _full_mask(target_tokens)
        elif self.args.noise == "no_noise":
            return target_tokens
        else:
            raise NotImplementedError
    
    def build_generator(self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None):
        # add models input to match the API for SequenceGenerator
        from fairseq.speech_iterative_refinement_generator import SpeechIterativeRefinementGenerator
        from fairseq.speech_assessment_generator import SpeechAssessmentGenerator
        from fairseq.speech_score_generator import SpeechScoreGenerator
        if getattr(args,'assessment',False):
            return SpeechAssessmentGenerator(
                self.target_dictionary,
                eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
                max_iter=getattr(args, "iter_decode_max_iter", 10),
                beam_size=getattr(args, "iter_decode_with_beam", 1),
                reranking=getattr(args, "iter_decode_with_external_reranker", False),
                decoding_format=getattr(args, "decoding_format", None),
                adaptive=not getattr(args, "iter_decode_force_max_iter", False),
                retain_history=getattr(args, "retain_iter_history", False),
                args=args,
            )
        elif getattr(args,'score',False):
            return SpeechScoreGenerator(
                self.target_dictionary,
                eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
                max_iter=getattr(args, "iter_decode_max_iter", 10),
                beam_size=getattr(args, "iter_decode_with_beam", 1),
                reranking=getattr(args, "iter_decode_with_external_reranker", False),
                decoding_format=getattr(args, "decoding_format", None),
                adaptive=not getattr(args, "iter_decode_force_max_iter", False),
                retain_history=getattr(args, "retain_iter_history", False),
            )
        else:
            return SpeechIterativeRefinementGenerator(
                self.target_dictionary,
                eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
                max_iter=getattr(args, "iter_decode_max_iter", 10),
                beam_size=getattr(args, "iter_decode_with_beam", 1),
                reranking=getattr(args, "iter_decode_with_external_reranker", False),
                decoding_format=getattr(args, "decoding_format", None),
                adaptive=not getattr(args, "iter_decode_force_max_iter", False),
                retain_history=getattr(args, "retain_iter_history", False),
            )
    
    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            # Though see Susanto et al. (ACL 2020): https://www.aclweb.org/anthology/2020.acl-main.325/
            raise NotImplementedError(
                "Constrained decoding with the translation_lev task is not supported"
            )

        return LanguagePairDataset(
            src_tokens, src_lengths, self.source_dictionary, append_bos=True
        )



    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        sample["prev_target"] = self.inject_noise(sample["target"])
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample["prev_target"] = self.inject_noise(sample["target"])
            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        if getattr(args,"no_bpe",False):
            return None
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

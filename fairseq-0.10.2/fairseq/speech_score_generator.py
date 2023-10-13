# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
from weakref import finalize

import numpy as np
import torch
from fairseq import utils
import json
from .iterative_refinement_generator import IterativeRefinementGenerator

DecoderOut = namedtuple(
    "IterativeRefinementDecoderOut",
    ["output_tokens", "output_scores", "attn", "decoder_out", "step", "max_step", "history"],
)


class SpeechScoreGenerator(IterativeRefinementGenerator):
    def __init__(
        self,
        tgt_dict,
        models=None,
        eos_penalty=0.0,
        max_iter=10,
        max_ratio=2,
        beam_size=1,
        decoding_format=None,
        retain_dropout=False,
        adaptive=True,
        retain_history=False,
        reranking=False,
    ):
        """
        Generates translations based on iterative refinement.

        Args:
            tgt_dict: target dictionary
            eos_penalty: if > 0.0, it penalized early-stopping in decoding
            max_iter: maximum number of refinement iterations
            max_ratio: generate sequences of maximum length ax, where x is the source length
            decoding_format: decoding mode in {'unigram', 'ensemble', 'vote', 'dp', 'bs'}
            retain_dropout: retaining dropout in the inference
            adaptive: decoding with early stop
        """
        super().__init__(
            tgt_dict,
            models,
            eos_penalty,
            max_iter,
            max_ratio,
            beam_size,
            decoding_format,
            retain_dropout,
            adaptive,
            retain_history,
            reranking)

    def generate_batched_itr(
        self,
        data_itr,
        maxlen_a=None,
        maxlen_b=None,
        cuda=False,
        timer=None,
        prefix_size=0,
    ):
        """Iterate over a batched dataset and yield individual translations.

        Args:
            maxlen_a/b: generate sequences of maximum length ax + b,
                where x is the source sentence length.
            cuda: use GPU for generation
            timer: StopwatchMeter for timing generations.
        """

        for sample in data_itr:
            if "net_input" not in sample:
                continue
            if timer is not None:
                timer.start()
            with torch.no_grad():
                hypos = self.generate(
                    self.models,
                    sample,
                    prefix_tokens=sample["target"][:, :prefix_size]
                    if prefix_size > 0
                    else None,
                )
            if timer is not None:
                timer.stop(sample["ntokens"])
            for i, id in enumerate(sample["id"]):
                # remove padding
                src = utils.strip_pad(sample["net_input"]["src_tokens"][i, :], self.pad)
                ref = utils.strip_pad(sample["target"][i, :], self.pad)
                yield id, src, ref, hypos[i]

    @torch.no_grad()
    def generate(self, models, sample, prefix_tokens=None, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the IterativeRefinementGenerator is not supported"
            )

        # TODO: iterative refinement generator does not support ensemble for now.
        if not self.retain_dropout:
            for model in models:
                model.eval()

        model, reranker = models[0], None
        if self.reranking:
            assert len(models) > 1, "Assuming the last checkpoint is the reranker"
            assert (
                self.beam_size > 1
            ), "Reranking requires multiple translation for each example"

            reranker = models[-1]
            models = models[:-1]

        if len(models) > 1 and hasattr(model, "enable_ensemble"):
            assert model.allow_ensemble, "{} does not support ensembling".format(
                model.__class__.__name__
            )
            model.enable_ensemble(models)

        # TODO: better encoder inputs?
        src_tokens = sample["net_input"]["src_tokens"]
        src_lengths = sample["net_input"]["src_lengths"]
        tgt_tokens = sample["target"]

        #label = sample["label"]

        bsz, src_len, _ = src_tokens.size()

        assert bsz==1, "score generator only support batch-size == 1."

        # initialize
        encoder_out = model.forward_encoder([src_tokens, src_lengths])

        tgt_lengs = tgt_tokens.ne(model.pad).sum(1)[0]
        
        finalized_tokens = torch.zeros(tgt_lengs).type_as(tgt_tokens)
        finalized_scores = torch.zeros(tgt_lengs).type_as(encoder_out.encoder_out)
        finalized_p = torch.zeros(tgt_lengs).type_as(encoder_out.encoder_out)
        
        
        initial_output_tokens = tgt_tokens.clone()
        initial_output_scores = initial_output_tokens.new_zeros(
            *initial_output_tokens.size()
        ).type_as(encoder_out.encoder_out)

        prev_decoder_out = DecoderOut(
            output_tokens=initial_output_tokens,
            output_scores=initial_output_scores,
            decoder_out = None,
            attn=None,
            step=0,
            max_step=0,
            history=None,
        )

        decoder_options = {
            "eos_penalty": self.eos_penalty,
            "max_ratio": self.max_ratio,
            "decoding_format": self.decoding_format,
        }

        decoder_out = model.forward_decoder(
            prev_decoder_out, encoder_out, **decoder_options
        )
        
        pred = decoder_out.decoder_out.squeeze()
        label = sample['label'].squeeze()
        
        loss = torch.nn.MSELoss()
        mse = loss(pred,label)

        def finalized_hypos(pred, label,mse):
            return {
                "pred":pred.tolist(),
                "label":label.tolist(),
                "mse":float(mse),
            }

        outpath='/home/v-yukliang/mycontainer/yukangliang/Projects/natasr/outputs/CMLM_score/output.txt'
        finalized=finalized_hypos(pred,label,mse)
        
        out=open(outpath,'a')
        out.write(json.dumps(finalized)+'\n')

        return finalized

    def rerank(self, reranker, finalized, encoder_input, beam_size):
        def rebuild_batch(finalized):
            finalized_tokens = [f[0]["tokens"] for f in finalized]
            finalized_maxlen = max(f.size(0) for f in finalized_tokens)
            final_output_tokens = (
                finalized_tokens[0]
                .new_zeros(len(finalized_tokens), finalized_maxlen)
                .fill_(self.pad)
            )
            for i, f in enumerate(finalized_tokens):
                final_output_tokens[i, : f.size(0)] = f
            return final_output_tokens

        final_output_tokens = rebuild_batch(finalized)
        final_output_tokens[
            :, 0
        ] = self.eos  # autoregressive model assumes starting with EOS

        reranker_encoder_out = reranker.encoder(*encoder_input)
        length_beam_order = (
            utils.new_arange(
                final_output_tokens, beam_size, reranker_encoder_out.encoder_out.size(1)
            )
            .t()
            .reshape(-1)
        )
        reranker_encoder_out = reranker.encoder.reorder_encoder_out(
            reranker_encoder_out, length_beam_order
        )
        reranking_scores = reranker.get_normalized_probs(
            reranker.decoder(final_output_tokens[:, :-1], reranker_encoder_out),
            True,
            None,
        )
        reranking_scores = reranking_scores.gather(2, final_output_tokens[:, 1:, None])
        reranking_masks = final_output_tokens[:, 1:].ne(self.pad)
        reranking_scores = (
            reranking_scores[:, :, 0].masked_fill_(~reranking_masks, 0).sum(1)
        )
        reranking_scores = reranking_scores / reranking_masks.sum(1).type_as(
            reranking_scores
        )

        for i in range(len(finalized)):
            finalized[i][0]["score"] = reranking_scores[i]

        return finalized

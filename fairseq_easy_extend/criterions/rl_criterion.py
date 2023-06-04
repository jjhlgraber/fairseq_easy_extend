import math
from argparse import Namespace

import torch
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import encoders
from fairseq.dataclass import FairseqDataclass

from dataclasses import dataclass, field

from fairseq.logging import metrics

# Added imports
from sacrebleu import sentence_bleu, sentence_chrf
import torch.nn.functional as F

# from torchtext.data.metrics import bleu_score
# import warnings

# with warnings.catch_warnings():
#     warnings.filterwarnings(
#         action="ignore",
#         category=UserWarning,
#     )
#     from torchmetrics.functional import chrf_score


@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(
        default="BLEU", metadata={"help": "sentence level metric"}
    )


@register_criterion("rl_loss", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric):
        super().__init__(task)
        self.metric = sentence_level_metric.lower()
        if self.metric == "bleu":
            self.metric_func = sentence_bleu
        elif self.metric == "chrf":
            self.metric_func = sentence_chrf
        else:
            raise Exception("RL metric not yet implemented")
        self.tokenizer = encoders.build_tokenizer(Namespace(tokenizer="moses"))
        self.tgt_dict = task.target_dictionary

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]
        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]
        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        # get loss only on tokens, not on lengths
        outs = outputs["word_ins"].get("out", None)
        masks = outputs["word_ins"].get("mask", None)

        loss, reward = self._compute_loss(outs, tgt_tokens, masks)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.detach(),
            "nll_loss": loss.detach(),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "reward": reward.detach(),
        }

        return loss, sample_size, logging_output

    def decode(self, toks, escape_unk=False):
        with torch.no_grad():
            s = self.tgt_dict.string(
                toks.int().cpu(),
                "@@ ",
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            s = self.tokenizer.decode(s)
        return s

    def compute_reward(self, sampled_sentences, targets):
        """
        #we take a softmax over outputs
        probs = F.softmax(outputs, dim=-1)
        #argmax over the softmax \ sampling (e.g. multinomial)
        samples_idx = torch.multinomial(probs, 1, replacement=True)
        sample_strings = self.tgt_dict.string(samples_idx)  #see dictionary class of fairseq
        #sample_strings = "I am a sentence"
        reward_vals = evaluate(sample_strings, targets)
        return reward_vals, samples_idx
        """
        with torch.no_grad():
            reward = torch.tensor(
                [
                    self.metric_func(sampled_sentence, [target]).score
                    for sampled_sentence, target in zip(sampled_sentences, targets)
                ]
            )
            return reward

    def _compute_loss(self, outputs, targets, masks=None):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """
        # padding mask
        ##If you take mask before you do sampling: you sample over a BATCH and your reward is on token level
        # if you take mask after, you sample SENTENCES and calculate reward on a sentence level
        # but make sure you apply padding mask after both on log prob outputs, reward and id's (you might need them for gather function to           extract log_probs of the samples)

        # Locate possible padding tokens for masking later
        masks = targets.ne(self.tgt_dict.pad())
        bsz, seq_len, vocab_size = outputs.size()
        # Flatten for sampling
        probs = F.softmax(outputs, dim=-1).view(-1, vocab_size)
        # Bring back to sentence view after sampling
        sample_idx = torch.multinomial(probs, 1, replacement=True).view(bsz, seq_len)

        sampled_sentences_strings = [
            self.decode(sample_idx_sent) for sample_idx_sent in sample_idx
        ]
        print(sampled_sentences_strings)
        targets_strings = [
            self.decode(
                target_sent,
            )
            for target_sent in targets
        ]
        print(targets_strings)

        with torch.no_grad():
            ####HERE calculate metric###
            reward = self.compute_reward(sampled_sentences_strings, targets_strings)

        # expand it to make it of a shape BxT - each token gets the same reward value (e.g. bleu is 20, so each token gets reward of 20 [20,20,20,20,20])
        reward = reward.unsqueeze(1).repeat(1, seq_len)

        # now you need to apply mask on both outputs and reward
        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]
            reward, sample_idx = reward[masks], sample_idx[masks]

        # numerically more stable than log on probs
        log_probs = F.log_softmax(outputs, dim=-1)

        loss = -log_probs.gather(1, sample_idx.unsqueeze(1)) * reward

        return loss.mean(), reward.mean()

        # Example 1: mask before sampling
        # if masks is not None:
        if True:
            outputs, targets = outputs[masks], targets[masks]
            # we take a softmax over outputs
            probs = F.softmax(outputs, dim=-1)

            # argmax over the softmax \ sampling (e.g. multinomial)
            sampled_sentence_idx = torch.multinomial(probs, 1, replacement=True)
            print(torch.sum(masks == False))
            print(
                self.decode(
                    sampled_sentence_idx,
                )
            )

            assert False

            with torch.no_grad():
                reward = self.compute_reward(sampled_sentences, targets)

                if self.metric == "bleu":
                    # Compute BLEU on token level
                    reward = torch.tensor(
                        [
                            bleu_score([sampled_sentence_token], [[targets_token]])
                            for sampled_sentence_token, targets_token in zip(
                                sampled_sentence_string.split(), targets_string.split()
                            )
                        ]
                    )
                elif self.metric == "chrf":
                    reward = chrf_score(
                        sampled_sentence_string.split(),
                        [[target_token] for target_token in targets_string.split()],
                        return_sentence_level_score=True,
                    )[1]
                else:
                    raise Exception("RL metric not yet implemented")
            # log_softmax on outputs again is numerically more stable
            loss = (
                -F.log_softmax(
                    outputs,
                    dim=-1,
                ).gather(1, sampled_sentence_idx)
                * reward
            )

        # Example 2: mask after sampling
        else:
            bsz, seq_len, vocab_size = outputs.size()
            # Flatten for sampling
            probs = F.softmax(outputs, dim=-1).view(-1, vocab_size)
            # Bring back to sentence view after sampling
            sample_idx = torch.multinomial(probs, 1, replacement=True).view(
                bsz, seq_len
            )

            sampled_sentences_strings = [
                self.decode(sample_idx_sent) for sample_idx_sent in sample_idx
            ]

            targets_strings = [
                self.decode(
                    target_sent,
                )
                for target_sent in targets
            ]

            # print(sampled_sentence_string) --> if you apply mask before, you get a sentence which is one token
            # imagine output[mask]=[MxV] where M is a sequence of all tokens in batch excluding padding symbols
            # now you sample 1 vocabulary index for each token, so you end up in [Mx1] matrix
            # when you apply string, it treats every token as a separate sentence --> hence you calc token-level metric. SO it makes much more sense to apply mask after sampling(!)
            with torch.no_grad():
                ####HERE calculate metric###

                if self.metric == "bleu":
                    # We follow the convention for comparibility of naively splitting on white space
                    # Compute the reward on sentence level
                    reward = torch.tensor(
                        [
                            bleu_score(
                                [sampled_sentence_string.split()],
                                [[target_string.split()]],
                            )
                            for sampled_sentence_string, target_string in zip(
                                sampled_sentences_strings, targets_strings
                            )
                        ]
                    )
                elif self.metric == "chrf":
                    reward = chrf_score(
                        sampled_sentences_strings,
                        targets_strings,
                        return_sentence_level_score=True,
                    )[1]
                else:
                    raise Exception("RL metric not yet implemented")

            # expand it to make it of a shape BxT - each token gets the same reward value (e.g. bleu is 20, so each token gets reward of 20 [20,20,20,20,20])
            reward = reward.unsqueeze(1).repeat(1, seq_len)

            # now you need to apply mask on both outputs and reward
            if masks is not None:
                outputs, targets = outputs[masks], targets[masks]
                reward, sample_idx = reward[masks], sample_idx[masks]

            loss = (
                -F.log_softmax(outputs, dim=-1).gather(1, sample_idx.unsqueeze(1))
                * reward
            )

        return loss.mean(), reward.mean()

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        reward_sum = sum(log.get("reward", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("reward", reward_sum / sample_size, sample_size, round=3)

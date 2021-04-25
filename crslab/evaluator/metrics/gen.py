# @Time   : 2020/11/30
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

# UPDATE:
# @Time   : 2020/12/18
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com

import re
from collections import Counter

import math
import numpy as np
from nltk import ngrams
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional

from crslab.evaluator.metrics.base import AverageMetric, SumMetric

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')
re_space = re.compile(r'\s+')


class PPLMetric(AverageMetric):
    def value(self):
        return math.exp(super().value())


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    s = re_space.sub(' ', s)
    # s = ' '.join(s.split())
    return s


class ExactMatchMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str]) -> 'ExactMatchMetric':
        if guess is None or answers is None:
            return None
        for a in answers:
            if guess == a:
                return ExactMatchMetric(1)
        return ExactMatchMetric(0)


class F1Metric(AverageMetric):
    """
    Helper class which computes token-level F1.
    """

    @staticmethod
    def _prec_recall_f1_score(pred_items, gold_items):
        """
        Compute precision, recall and f1 given a set of gold and prediction items.

        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values

        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    @staticmethod
    def compute(guess: str, answers: List[str]) -> 'F1Metric':
        if guess is None or answers is None:
            return AverageMetric(0, 0)
        g_tokens = guess.split()
        scores = [
            F1Metric._prec_recall_f1_score(g_tokens, a.split())
            for a in answers
        ]
        return F1Metric(max(scores), 1)


class BleuMetric(AverageMetric):
    @staticmethod
    def compute(guess: str, answers: List[str], k: int) -> Optional['BleuMetric']:
        """
        Compute approximate BLEU score between guess and a set of answers.
        """

        weights = [0] * 4
        weights[k - 1] = 1
        score = sentence_bleu(
            [a.split(" ") for a in answers],
            guess.split(" "),
            weights=weights,
        )
        return BleuMetric(score)


class DistMetric(SumMetric):
    @staticmethod
    def compute(sent: str, k: int) -> 'DistMetric':
        token_set = set()
        for token in ngrams(sent.split(), k):
            token_set.add(token)
        return DistMetric(len(token_set))


class EmbeddingAverage(AverageMetric):
    @staticmethod
    def _avg_embedding(embedding):
        return np.sum(embedding, axis=0) / (np.linalg.norm(np.sum(embedding, axis=0)) + 1e-12)

    @staticmethod
    def compute(hyp_embedding, ref_embeddings) -> 'EmbeddingAverage':
        hyp_avg_emb = EmbeddingAverage._avg_embedding(hyp_embedding).reshape(1, -1)
        ref_avg_embs = [EmbeddingAverage._avg_embedding(emb) for emb in ref_embeddings]
        ref_avg_embs = np.array(ref_avg_embs)
        return EmbeddingAverage(float(cosine_similarity(hyp_avg_emb, ref_avg_embs).max()))


class VectorExtrema(AverageMetric):
    @staticmethod
    def _extreme_embedding(embedding):
        max_emb = np.max(embedding, axis=0)
        min_emb = np.min(embedding, axis=0)
        extreme_emb = np.fromiter(
            map(lambda x, y: x if ((x > y or x < -y) and y > 0) or ((x < y or x > -y) and y < 0) else y, max_emb,
                min_emb), dtype=float)
        return extreme_emb

    @staticmethod
    def compute(hyp_embedding, ref_embeddings) -> 'VectorExtrema':
        hyp_ext_emb = VectorExtrema._extreme_embedding(hyp_embedding).reshape(1, -1)
        ref_ext_embs = [VectorExtrema._extreme_embedding(emb) for emb in ref_embeddings]
        ref_ext_embs = np.asarray(ref_ext_embs)
        return VectorExtrema(float(cosine_similarity(hyp_ext_emb, ref_ext_embs).max()))


class GreedyMatch(AverageMetric):
    @staticmethod
    def compute(hyp_embedding, ref_embeddings) -> 'GreedyMatch':
        hyp_emb = np.asarray(hyp_embedding)
        ref_embs = (np.asarray(ref_embedding) for ref_embedding in ref_embeddings)
        score_max = 0
        for ref_emb in ref_embs:
            sim_mat = cosine_similarity(hyp_emb, ref_emb)
            score_max = max(score_max, (sim_mat.max(axis=0).mean() + sim_mat.max(axis=1).mean()) / 2)
        return GreedyMatch(score_max)

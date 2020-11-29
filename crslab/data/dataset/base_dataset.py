# @Time   : 2020/11/22
# @Author : Kun Zhou
# @Email  : francis_kun_zhou@163.com

# UPDATE:
# @Time   : 2020/11/23, 2020/11/29
# @Author : Kun Zhou, Xiaolei Wang
# @Email  : francis_kun_zhou@163.com, wxl1999@foxmail.com

import os
import pickle as pkl
from abc import ABC, abstractmethod
from os.path import dirname, realpath

import numpy as np
from gensim.models import Word2Vec, FastText
from loguru import logger
from torchtext import vocab

ROOT_PATH = dirname(dirname(dirname(dirname(realpath(__file__)))))
DATA_PATH = os.path.join(ROOT_PATH, "data")


class BaseDataset(ABC):
    """:class:`Dataset` stores the original dataset in memory.
    It provides many useful functions for data preprocessing. Finally, the dataset are preprocessed as
    {
        'dialog_context': the preprocessed contextual dialog;
        'interaction_context': if necessary, the preprocessed interaction history;
        'entity_context': if necessary, the entities in context;
        'word_context': if necessary, the words in context;
        'rec_item': the recommended item in this turn;
        'response': the ground-truth response;
        'guiding_thread': the guiding topic;
        'entity_knowledge_graph': if necessary, entity knowledge graph as side information;
        'word_knowledge_graph': if necessary, word knowledge graph as side information;
    }

    Args:
        opt (Config): Global configuration object.

    Attributes:
    """

    def __init__(self, opt, dpath, restore=False, save=False):
        self.opt = opt
        self.dpath = dpath

        if not restore:
            train_data, valid_data, test_data, self.tok2ind, self.ind2tok = self._load_data()
            self.train_data, self.valid_data, self.test_data, self.side_data = self._data_preprocess(train_data,
                                                                                                     valid_data,
                                                                                                     test_data)
            embedding = self._pretrain_embedding(train_data)
            if embedding:
                self.side_data["embedding"] = embedding
        else:
            self.train_data, self.valid_data, self.test_data, self.side_data, self.tok2ind, self.ind2tok = self._load_from_restore()

        if save:
            data = (self.train_data, self.valid_data, self.test_data, self.side_data, self.tok2ind, self.ind2tok)
            self._save_to_one(data)

    @abstractmethod
    def _load_data(self):
        """return train, valid, test data and tok2ind, ind2tok"""
        raise NotImplementedError

    @abstractmethod
    def _data_preprocess(self, train_data, valid_data, test_data):
        """return train, valid, test, side data"""
        raise NotImplementedError

    def _load_from_restore(self, file_name="all_data.pkl"):
        """Restore saved dataset from ``saved_dataset``.

        Args:
            file_name (str): file for the saved dataset.
        """
        logger.debug('Restoring dataset from [{}]'.format(file_name))
        if not os.path.exists(os.path.join(self.dpath, file_name)):
            raise ValueError('filepath [{}] does not exist'.format(file_name))
        with open(os.path.join(self.dpath, file_name), 'rb') as f:
            return pkl.load(f)

    def _save_to_one(self, data, file_name="all_data.pkl"):
        save_path = os.path.join(self.dpath, file_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(save_path, 'wb') as f:
            pkl.dump(data, f)

    @staticmethod
    def add_start_end_token_idx(vec: list, add_start=False, start_idx=None, add_end=False, end_idx=None):
        if add_start:
            vec.insert(0, start_idx)
        if add_end:
            vec.append(end_idx)
        return vec

    @staticmethod
    def truncate(vec, max_length, truncate_tail=True):
        """Check that vector is truncated correctly."""
        if max_length is None:
            return vec
        if len(vec) <= max_length:
            return vec
        if truncate_tail:
            return vec[:max_length]
        else:
            return vec[-max_length:]

    def _pretrain_embedding(self, data=None):
        emb_type = self.opt.get('embedding_type', None)
        if emb_type is None:
            return None

        embedding = np.zeros(len(self.tok2ind), self.opt["embedding_dim"])
        pretrained = self.opt.get("pretrained", False)
        dim = self.opt.get('embedding_dim', 300)
        if pretrained:
            if emb_type == "fasttext":
                name = 'fasttext'
                model = vocab.FastText(cache=os.path.join(self.dpath, "fasttext_vectors"))
            elif emb_type == "fasttext_cc":
                name = 'fasttext_cc'
                model = vocab.Vectors(
                    name='crawl-300d-2M.vec',
                    url='https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip',
                    cache=os.path.join(self.dpath, 'fasttext_cc_vectors'),
                )
            elif emb_type == "glove":
                name = self.opt.get('embedding_name', '840B')
                model = vocab.GloVe(name=name, dim=dim, cache=os.path.join(self.dpath, 'glove_vectors'))
                name = 'glove_' + name
            else:
                raise RuntimeError(
                    'embedding type {} not implemented. check arg, '
                    'submit PR to this function, or override it.'
                    ''.format(emb_type)
                )

            cnt = 0
            for w, i in self.tok2ind.items():
                if w in model.stoi:
                    embedding[i] = model.vectors[model.stoi[w]]
                    cnt += 1
            logger.info(
                f'[Initialized embeddings for {cnt} tokens '
                f'({cnt / len(self.tok2ind):.1%}) from {name}_{dim}d]'
            )
        else:
            def corpus():
                for conv in data:
                    for utt in conv["dialog"]:
                        yield utt["text"]

            logger.info(f'[Start pretrain {emb_type}]')
            opt = self.opt.get(emb_type, None)
            if emb_type == "word2vec":
                if opt:
                    model = Word2Vec(corpus, size=dim, **opt)
                else:
                    model = Word2Vec(corpus, size=dim)
            elif emb_type == "fasttext":
                if opt:
                    model = FastText(corpus, size=dim, **self.opt['fasttext'])
                else:
                    model = FastText(corpus, size=dim)
            else:
                raise RuntimeError(
                    'embedding type {} not implemented. check arg, '
                    'submit PR to this function, or override it.'
                    ''.format(emb_type)
                )
            for w, i in self.tok2ind.items():
                if w in model:
                    embedding[i] = model[w]
            logger.info(f'[Finish pretrain {emb_type}]')

        return embedding

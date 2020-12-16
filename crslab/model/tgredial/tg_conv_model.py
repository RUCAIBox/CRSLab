# @Time   : 2020/12/9
# @Author : Yuanhang Zhou
# @Email  : sdzyh002@gmail.com

# UPDATE:
# @Time   : 2020/12/16
# @Author : Xiaolei Wang
# @Email  : wxl1999@foxmail.com
import os

import torch
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel

from crslab.config.config import MODEL_PATH
from crslab.model.base_model import BaseModel
from .resource import resources


class TGConvModel(BaseModel):
    def __init__(self, opt, device, vocab, side_data):
        self.context_truncate = opt['context_truncate']
        self.response_truncate = opt['response_truncate']
        self.pad_id = vocab['pad']

        dataset = opt['dataset']
        dpath = os.path.join(MODEL_PATH, "tgredial", dataset)
        resource = resources[dataset]
        super(TGConvModel, self).__init__(opt, device, dpath, resource)

    def build_model(self):
        """build model"""
        # model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(
        #     self.model_config)
        # self.model = GPT2LMHeadModel(config=model_config)
        self.model = GPT2LMHeadModel.from_pretrained(os.path.join(self.dpath, 'gpt2'))

        # self.model.resize_token_embeddings(self.vocab_size)
        # self.n_ctx = self.model.config.to_dict().get("n_ctx")  # not used

        self.loss = CrossEntropyLoss(ignore_index=self.pad_id)

    def forward(self, batch, mode):
        input_ids, context, _, _, y = batch

        if mode != 'test':
            # torch.tensor's shape = (bs, seq_len, v_s); tuple's length = 12
            lm_logits = self.model(input_ids).logits

            # index from 1 to self.reponse_truncate is valid response
            loss = self.calculate_loss(
                lm_logits[:, -self.response_truncate:-1, :],
                input_ids[:, -self.response_truncate + 1:])

            pred = torch.max(lm_logits, dim=2)[1]  # [bs, seq_len]
            pred = pred[:, -self.response_truncate:]

            return loss, pred
        else:
            return self.generate(context)

    def generate(self, context):
        """
        Args:
            context: torch.tensor, shape=(bs, context_turncate)

        Returns:
            generated_response: torch.tensor, shape=(bs, reponse_turncate-1)
        """
        generated_response = []
        former_hidden_state = None
        context = context[..., -self.response_truncate + 1:]

        for i in range(self.response_truncate - 1):
            outputs = self.model(context, former_hidden_state)  # (bs, c_t, v_s),
            last_hidden_state, former_hidden_state = outputs.hidden_states, outputs.past_key_values

            next_token_logits = last_hidden_state[-1][:, -1, :]  # (bs, v_s)
            preds = next_token_logits.argmax(dim=-1).long()  # (bs)

            context = preds.unsqueeze(1)
            generated_response.append(preds)

        generated_response = torch.stack(generated_response).T

        return generated_response

    def calculate_loss(self, logit, labels):
        """
        Args:
            preds: torch.FloatTensor, shape=(bs, response_truncate, vocab_size)
            labels: torch.LongTensor, shape=(bs, response_truncate)

        """

        loss = self.loss(logit.reshape(-1, logit.size(-1)), labels.reshape(-1))
        return loss

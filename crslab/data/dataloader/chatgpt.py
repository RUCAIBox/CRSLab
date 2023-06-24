# @Time   : 2023/6/14
# @Author : Xinyu Tang
# @Email  : txy20010310@163.com

from tqdm import tqdm

from crslab.data.dataloader.base import BaseDataLoader

class ChatGPTDataLoader(BaseDataLoader):
    
    def __init__(self, opt, dataset, vocab):
        super().__init__(opt, dataset)
        
    def process_fn(self):
        augment_dataset = []
        for conv_dict in tqdm(self.dataset):
            if conv_dict['role'] == 'Recommender' and len(conv_dict['items']) > 0:
                augment_conv_dict = {
                    'dialog_id': conv_dict['dialog_id'],
                    'role': conv_dict['role'],
                    'entity': conv_dict['context_entities'],
                    'context': conv_dict['context'],
                    'item': conv_dict['items']
                }
                augment_dataset.append(augment_conv_dict)
        return augment_dataset
    
    def batchify(self, batch):
        batch_dialog_id = []
        batch_role = []
        batch_context = []
        batch_movies = []
        batch_entities = []
        
        for conv_dict in batch:
            batch_dialog_id.append(conv_dict['dialog_id'])
            batch_role.append(conv_dict['role'])
            batch_context.append(conv_dict['context'])
            batch_movies.append(conv_dict['item'])
            batch_entities.append(conv_dict['entity'])

        return {
            'dialog_id': batch_dialog_id,
            'role': batch_role,
            'context': batch_context,
            'item': batch_movies,
            'entity': batch_entities
        }
    
    def rec_process_fn(self):
        return self.process_fn()
    
    def rec_batchify(self, batch):
        return self.rec_batchify(batch)
    
    def conv_process_fn(self):
        return self.process_fn()
    
    def conv_batchify(self, batch):
        return self.batchify(batch)
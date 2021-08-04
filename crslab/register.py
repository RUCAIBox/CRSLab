# @Time   : 2021/8/4
# @Author : Chenzhan Shang
# @Email  : czshang@outlook.com


from crslab.dataset.durecdial import DuRecDialDataset
from crslab.dataset.gorecdial import GoRecDialDataset
from crslab.dataset.inspired import InspiredDataset
from crslab.dataset.opendialkg import OpenDialKGDataset
from crslab.dataset.redial import ReDialDataset
from crslab.dataset.tgredial import TGReDialDataset
from crslab.dataset.lastfm import LastFMDataset

from crslab.agent.supervised import *
from crslab.agent.interactive import *

from crslab.evaluator.conv import ConvEvaluator
from crslab.evaluator.rec import RecEvaluator
from crslab.evaluator.standard import StandardEvaluator

from crslab.model.conversation import *
from crslab.model.crs import *
from crslab.model.policy import *
from crslab.model.recommendation import *

from crslab.system.inspired import InspiredSystem
from crslab.system.kbrd import KBRDSystem
from crslab.system.kgsf import KGSFSystem
from crslab.system.redial import ReDialSystem
from crslab.system.tgredial import TGReDialSystem


dataset_register_table = {
    'ReDial': ReDialDataset,
    'TGReDial': TGReDialDataset,
    'GoRecDial': GoRecDialDataset,
    'OpenDialKG': OpenDialKGDataset,
    'Inspired': InspiredDataset,
    'DuRecDial': DuRecDialDataset,
    'LastFM': LastFMDataset
}

dataset_language_map = {
    'ReDial': 'en',
    'TGReDial': 'zh',
    'GoRecDial': 'en',
    'OpenDialKG': 'en',
    'Inspired': 'en',
    'DuRecDial': 'zh'
}

agent_register_table = {
    'KGSF': KGSFAgent,
    'KBRD': KBRDAgent,
    'TGReDial': TGReDialAgent,
    'TGRec': TGReDialAgent,
    'TGConv': TGReDialAgent,
    'TGPolicy': TGReDialAgent,
    'TGRec_TGConv': TGReDialAgent,
    'TGRec_TGConv_TGPolicy': TGReDialAgent,
    'ReDialRec': ReDialAgent,
    'ReDialConv': ReDialAgent,
    'ReDialRec_ReDialConv': ReDialAgent,
    'InspiredRec_InspiredConv': InspiredAgent,
    'BERT': TGReDialAgent,
    'SASREC': TGReDialAgent,
    'TextCNN': TGReDialAgent,
    'GRU4REC': TGReDialAgent,
    'Popularity': TGReDialAgent,
    'Transformer': KGSFAgent,
    'GPT2': TGReDialAgent,
    'ConvBERT': TGReDialAgent,
    'TopicBERT': TGReDialAgent,
    'ProfileBERT': TGReDialAgent,
    'MGCG': TGReDialAgent,
    'PMI': TGReDialAgent,
    'SCPR': SCPRAgent,
    'EAR': EARAgent
}

evaluator_register_table = {
    'rec': RecEvaluator,
    'conv': ConvEvaluator,
    'standard': StandardEvaluator
}

model_register_table = {
    'KGSF': KGSFModel,
    'KBRD': KBRDModel,
    'TGRec': TGRecModel,
    'TGConv': TGConvModel,
    'TGPolicy': TGPolicyModel,
    'ReDialRec': ReDialRecModel,
    'ReDialConv': ReDialConvModel,
    'InspiredRec': InspiredRecModel,
    'InspiredConv': InspiredConvModel,
    'GPT2': GPT2Model,
    'Transformer': TransformerModel,
    'ConvBERT': ConvBERTModel,
    'ProfileBERT': ProfileBERTModel,
    'TopicBERT': TopicBERTModel,
    'PMI': PMIModel,
    'MGCG': MGCGModel,
    'BERT': BERTModel,
    'SASREC': SASRECModel,
    'GRU4REC': GRU4RECModel,
    'Popularity': PopularityModel,
    'TextCNN': TextCNNModel
}

system_register_table = {
    'ReDialRec_ReDialConv': ReDialSystem,
    'KBRD': KBRDSystem,
    'KGSF': KGSFSystem,
    'TGRec_TGConv': TGReDialSystem,
    'TGRec_TGConv_TGPolicy': TGReDialSystem,
    'InspiredRec_InspiredConv': InspiredSystem,
    'GPT2': TGReDialSystem,
    'Transformer': TGReDialSystem,
    'ConvBERT': TGReDialSystem,
    'ProfileBERT': TGReDialSystem,
    'TopicBERT': TGReDialSystem,
    'PMI': TGReDialSystem,
    'MGCG': TGReDialSystem,
    'BERT': TGReDialSystem,
    'SASREC': TGReDialSystem,
    'GRU4REC': TGReDialSystem,
    'Popularity': TGReDialSystem,
    'TextCNN': TGReDialSystem
}

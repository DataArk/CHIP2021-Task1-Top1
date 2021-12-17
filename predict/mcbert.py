import os
import jieba
import torch
import pickle
import torch.nn as nn
import torch.optim as optim

from ark_nlp.dataset import TMDataset
from ark_nlp.processor.vocab import CharVocab
from ark_nlp.processor.tokenizer.tm import TransfomerTokenizer
from ark_nlp.nn import Bert
from ark_nlp.dataset import BaseDataset

import pandas as pd


import codecs
import json
import os

import random
import torch
import random
import numpy as np


def set_seed(seed):
    """
    设置随机种子
    :param seed:
    
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
set_seed(2021)


import copy
import torch
import pandas as pd

from functools import lru_cache
from torch.utils.data import Dataset
from ark_nlp.dataset import PairSentenceClassificationDataset


class TMDataset(PairSentenceClassificationDataset):
    def __init__(self, *args, **kwargs):
        
        super(TMDataset, self).__init__(*args, **kwargs)
        self.categories_b = sorted(list(set([data['label_b'] for data in self.dataset])))
        self.cat2id_b = dict(zip(self.categories_b, range(len(self.categories_b))))
        self.id2cat_b = dict(zip(range(len(self.categories_b)), self.categories_b))
        
    def _convert_to_transfomer_ids(self, bert_tokenizer):
        
        features = []
        for (index_, row_) in enumerate(self.dataset):
            input_ids = bert_tokenizer.sequence_to_ids(row_['text_a'], row_['text_b'])
            
            input_ids, input_mask, segment_ids, speaker_ids, e1_mask = input_ids
                        
            input_a_length = self._get_input_length(row_['text_a'], bert_tokenizer)
            input_b_length = self._get_input_length(row_['text_b'], bert_tokenizer)

            feature = {
                'input_ids': input_ids, 
                'attention_mask': input_mask, 
                'token_type_ids': segment_ids,
                'speaker_ids': speaker_ids,
                'e1_mask': e1_mask
            }

            if not self.is_test:
                label_ids = self.cat2id[row_['label']]
                label_ids_b = self.cat2id_b[row_['label_b']]

                feature['label_ids'] = label_ids
                feature['label_ids_b'] = label_ids_b

            features.append(feature)
        
        return features  
    
    
import numpy as np
import pandas as pd
import copy

# from utils import get_entity_bios
from ark_nlp.dataset import BaseDataset


def get_task_data(data_path):
    with codecs.open(data_path, mode='r', encoding='utf8') as f:
        reader = f.readlines(f)    
        
    data_list = []

    for dialogue_ in reader:
        dialogue_ = json.loads(dialogue_)
        
        _dialog_id = dialogue_['dialog_id']
        
        for content_idx_, contents_ in enumerate(dialogue_['dialog_info']):

            terms_ = contents_['ner']

            if len(terms_) != 0:
                idx_ = 0
                for _, term_ in enumerate(terms_):
                    
                    entity_ = dict()

                    entity_['dialogue'] = dialogue_
                    
                    _text = dialogue_['dialog_info'][content_idx_]['text']
                    _text_list = list(_text)
                    _text_list.insert(term_['range'][0], '[unused1]')
                    _text_list.insert(term_['range'][1]+1, '[unused2]')
                    _text = ''.join(_text_list)
                    
                    if content_idx_ - 1 >= 0 and len(dialogue_['dialog_info'][content_idx_-1]) < 40:
                        forward_text = dialogue_['dialog_info'][content_idx_-1]['sender'] + ':' + dialogue_['dialog_info'][content_idx_-1]['text'] + ';'
                    else:
                        forward_text = ''
                    
                    if contents_['sender'] == '医生':
                        
                        if content_idx_ + 1 >= len(dialogue_['dialog_info']):
                            entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text
                        else:
                            entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text + ';'
                            temp_index = copy.deepcopy(content_idx_) + 1
                        
                            speaker_flag = False
                            sen_counter = 0
                            while True:

                                if dialogue_['dialog_info'][temp_index]['sender'] == '患者':
                                    sen_counter += 1
                                    speaker_flag = True
                                    entity_['text_a'] += dialogue_['dialog_info'][temp_index]['sender'] + ':' + dialogue_['dialog_info'][temp_index]['text'] + ';'
                                
                                if sen_counter > 3:
                                    break
                                
                                temp_index += 1
                                if temp_index >= len(dialogue_['dialog_info']):
                                    break
                                    
                    elif contents_['sender'] == '患者':
                        if content_idx_ + 1 >= len(dialogue_['dialog_info']):
                            entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text
                        else:
                            entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text + ';'
                            temp_index = copy.deepcopy(content_idx_) + 1
                        
                            speaker_flag = False
                            sen_counter = 0
                            while True:

                                sen_counter += 1
                                speaker_flag = True
                                entity_['text_a'] += dialogue_['dialog_info'][temp_index]['sender'] + ':' + dialogue_['dialog_info'][temp_index]['text'] + ';'
                                
                                if sen_counter > 3:
                                    break
                                
                                temp_index += 1
                                if temp_index >= len(dialogue_['dialog_info']):
                                    break
                    else:
                        entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text
                        
                    if term_['name'] == 'undefined':
                        add_text = '|没有标准化'
                    else:
                        add_text = '|标准化为' + term_['name']
                        
                    entity_['text_b'] = term_['mention'] + add_text
                    entity_['text_b_copy'] = term_['mention'] 
                    entity_['start_idx'] = term_['range'][0]
                    entity_['end_idx'] = term_['range'][1] - 1
                    
                    try:
                        entity_['label_b'] = term_['name']
                    except:
                        print(contents_)
                        print(term_)
                    entity_['label'] = term_['attr']
                    entity_['dialog_id'] = _dialog_id
                    idx_ += 1
                    
                    if entity_['label'] == '':
                        continue
                    
                    if len(entity_) == 0:
                        continue
                        
                    data_list.append(entity_)
                
            
    data_df = pd.DataFrame(data_list)
    
    
    data_df = data_df.loc[:,['dialog_id', 'text_b_copy', 'text_a', 'text_b', 'start_idx', 'end_idx', 'label_b', 'label', 'dialogue']]
    
    return data_df


import re
import copy


data_df = get_task_data('../data/source_datasets/fliter_train_result2.txt')


tm_dataset = TMDataset(data_df)


import transformers 
from transformers import AutoTokenizer

bert_vocab = transformers.AutoTokenizer.from_pretrained('../pretrained_model/mc_bert')


import unicodedata
import abc
import torch
import random
import transformers 
import numpy as np

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from copy import deepcopy
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from ark_nlp.processor.tokenizer._tokenizer import BaseTokenizer


class TransfomerTokenizer(BaseTokenizer):
    """
    Transfomer文本编码器，用于对文本进行分词、ID化、填充等操作

    :param max_seq_len: (int) 预设的文本最大长度
    :param tokenizer: (object) 编码器，用于实现文本分词和ID化

    """
    def __init__(self, vocab, max_seq_len):

        if isinstance(vocab, str):
            # TODO: 改成由自定义的字典所决定
            vocab = transformers.AutoTokenizer.from_pretrained(vocab)

        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.additional_special_tokens = set()
        self.tokenizer_type = 'transfomer'

    @staticmethod
    def _is_control(ch):
        """控制类字符判断
        """
        return unicodedata.category(ch) in ('Cc', 'Cf')

    @staticmethod
    def _is_special(ch):
        """判断是不是有特殊含义的符号
        """
        return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')

    @staticmethod
    def recover_bert_token(token):
        """获取token的“词干”（如果是##开头，则自动去掉##）
        """
        if token[:2] == '##':
            return token[2:]
        else:
            return token

    def get_token_mapping(self, text, tokens, is_mapping_index=True):
        """给出原始的text和tokenize后的tokens的映射关系
        """
        raw_text = deepcopy(text)
        text = text.lower()

        normalized_text, char_mapping = '', []
        for i, ch in enumerate(text):
            ch = unicodedata.normalize('NFD', ch)
            ch = ''.join([c for c in ch if unicodedata.category(c) != 'Mn'])
            ch = ''.join([
                c for c in ch
                if not (ord(c) == 0 or ord(c) == 0xfffd or self._is_control(c))
            ])
            normalized_text += ch
            char_mapping.extend([i] * len(ch))

        text, token_mapping, offset = normalized_text, [], 0
        for token in tokens:
            token = token.lower()
            if token == '[unk]' or token in self.additional_special_tokens:
                if is_mapping_index:
                    token_mapping.append(char_mapping[offset:offset+1])
                else:
                    token_mapping.append(raw_text[offset:offset+1])
                offset = offset + 1
            elif self._is_special(token):
                token_mapping.append([]) # 如果是[CLS]或者是[SEP]之类的词，则没有对应的映射
            else:
                token = self.recover_bert_token(token)
                start = text[offset:].index(token) + offset
                end = start + len(token)
                if is_mapping_index:
                    token_mapping.append(char_mapping[start:end])
                else:
                    token_mapping.append(raw_text[start:end])
                offset = end

        return token_mapping

    def sequence_to_ids(self, sequence_a, sequence_b=None):
        if sequence_b is None:
            return self.sentence_to_ids(sequence_a)
        else:
            return self.pair_to_ids(sequence_a, sequence_b)

    def sentence_to_ids(self, sequence, return_sequence_length=False):
        if type(sequence) == str:
            sequence = self.tokenize(sequence) 

        if return_sequence_length:
            sequence_length = len(sequence)

        # 对超长序列进行截断
        if len(sequence) > self.max_seq_len - 2:
            sequence = sequence[0:(self.max_seq_len - 2)]
        
        speaker_ids = []
        id_ = 0
        for idx_, term_ in enumerate(sequence):
            if term_ == '医' and sequence[idx_+1] == '生':
                id_ = 1
            if term_ == '患' and sequence[idx_+1] == '者':
                id_ = 2
            
            speaker_ids.append(id_)
                            
        # 分别在首尾拼接特殊符号
        sequence = ['[CLS]'] + sequence + ['[SEP]'] 
        speaker_ids = [0] + speaker_ids + [0]
        segment_ids = [0] * len(sequence) 
        
        e11_p = sequence.index("<") + 1
        e12_p = sequence.index(">") - 1
        
        e1_mask = [0] * len(sequence) 
        for _i in range(e11_p, e12_p+1):
            e1_mask[_i] = 1
            break
        
        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)

        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(sequence))
        # 创建seq_mask
        sequence_mask = [1] * len(sequence) + padding
        # 创建seq_segment
        segment_ids = segment_ids + padding
        # 对seq拼接填充序列
        sequence += padding
        e1_mask += padding

        sequence = np.asarray(sequence, dtype='int64')
        sequence_mask = np.asarray(sequence_mask, dtype='int64')
        segment_ids = np.asarray(segment_ids, dtype='int64')
        e1_mask = np.asarray(e1_mask, dtype='int64')

        if return_sequence_length:
            return (sequence, sequence_mask, segment_ids, e1_mask, sequence_length)

        return (sequence, sequence_mask, segment_ids, e1_mask)

    def pair_to_ids(self, sequence_a, sequence_b, return_sequence_length=False):
        
        raw_sequence_a = copy.deepcopy(sequence_a)
        
        if type(sequence_a) == str:
            sequence_a = self.tokenize(sequence_a)

        if type(sequence_b) == str:
            sequence_b = self.tokenize(sequence_b) 

        if return_sequence_length:
            sequence_length = (len(sequence_a), len(sequence_b))

        # 对超长序列进行截断
        start_idx = 0 
        end_idx = self.max_seq_len - len(sequence_b) - 3
        entity_end_idx = sequence_a.index('[unused2]')
        end_idx = entity_end_idx + 20
        if end_idx <  (self.max_seq_len - len(sequence_b)):
            sequence_a = sequence_a[0:(self.max_seq_len - len(sequence_b))- 3]
        else:
            end_idx = end_idx - 20 + (self.max_seq_len - len(sequence_b))/ 2
            start_idx = end_idx - (self.max_seq_len - len(sequence_b)) + 3
            if start_idx < 0:
                start_idx = 0
            sequence_a = sequence_a[int(start_idx):int(end_idx)]
        
        
        
#         sequence_a = sequence_a[0:(self.max_seq_len - len(sequence_b))]
#         if len(sequence_a) > ((self.max_seq_len - 3)//2):
#             sequence_a = sequence_a[0:(self.max_seq_len - 3)//2]
#         if len(sequence_b) > ((self.max_seq_len - 3)//2):
#             sequence_b = sequence_b[0:(self.max_seq_len - 3)//2]
            
        speaker_ids = [0]
        id_ = 0
        for idx_, term_ in enumerate(sequence_a):
            try:
                if term_ == '医' and idx_ < len(sequence_a) - 1 and sequence_a[idx_+1] == '生':
                    id_ = 1
                if term_ == '患' and idx_ < len(sequence_a) - 1 and sequence_a[idx_+1] == '者':
                    id_ = 2
            except:
                print(sequence_a)
                print(idx_)
            
            speaker_ids.append(id_)
        
        speaker_ids.append(0)
        for idx_, term_ in enumerate(sequence_b):
            
            speaker_ids.append(3)
        speaker_ids.append(0)
                

        # 分别在首尾拼接特殊符号
        sequence = ['[CLS]'] + sequence_a + ['[SEP]'] + sequence_b + ['[SEP]']
        segment_ids = [0] * (len(sequence_a) + 2) + [1] * (len(sequence_b) + 1)
        try:
            e11_p = sequence.index("[unused1]") + 1
            e12_p = sequence.index("[unused2]") - 1
        except:
            print(raw_sequence_a)
            print(sequence_a)
        
        e1_mask = [0] * len(sequence) 
        for _i in range(e11_p, e12_p+1):
            e1_mask[_i] = 1

        # ID化
        sequence = self.vocab.convert_tokens_to_ids(sequence)

        # 根据max_seq_len与seq的长度产生填充序列
        padding = [0] * (self.max_seq_len - len(sequence))
        # 创建seq_mask
        sequence_mask = [1] * len(sequence) + padding
        # 创建seq_segment
        segment_ids = segment_ids + padding
        # 对seq拼接填充序列
        sequence += padding
        
        speaker_ids += padding
        e1_mask += padding

        sequence = np.asarray(sequence, dtype='int64')
        sequence_mask = np.asarray(sequence_mask, dtype='int64')
        segment_ids = np.asarray(segment_ids, dtype='int64')
        speaker_ids = np.asarray(speaker_ids, dtype='int64')
        e1_mask = np.asarray(e1_mask, dtype='int64')
        
#         if len(sequence) > 150:
#             print('sequence', raw_sequence_a)
#         if len(sequence_mask) > 150:
#             print(len(sequence_mask))
#             print(len(sequence))
#             print('sequence_mask', raw_sequence_a)
#         if len(segment_ids) > 150:
#             print('segment_ids', raw_sequence_a)
#         if len(speaker_ids) > 150:
#             print('speaker_ids', raw_sequence_a)
#         if len(e1_mask) > 150:
#             print('e1_mask', raw_sequence_a)            
            

        if return_sequence_length:
            return (sequence, sequence_mask, segment_ids, speaker_ids, e1_mask, sequence_length)

        return (sequence, sequence_mask, segment_ids, speaker_ids, e1_mask)
    
bert_vocab.add_special_tokens({'additional_special_tokens':["[unused1]", "[unused2]", "|"]})


max_seq_length=200

tokenizer = TransfomerTokenizer(bert_vocab, max_seq_length)

import time
import torch
import math
import torch.nn.functional as F

from torch import nn
from torch import Tensor
from ark_nlp.nn import BasicModule
from transformers import BertModel
from transformers import BertPreTrainedModel
from torch.nn import CrossEntropyLoss
from ark_nlp.nn.layer.crf_block import CRF


class Bert(BertPreTrainedModel):
    """
    原始的BERT模型

    :param config: (obejct) 模型的配置对象
    :param bert_trained: (bool) bert参数是否可训练，默认可训练

    :returns: 

    Reference:
        [1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  
    """ 
    def __init__(
        self, 
        config, 
        encoder_trained=True,
        pooling='cls'
    ):
        super(Bert, self).__init__(config)
        
        self.bert = BertModel(config)
        self.pooling = pooling
        
        for param in self.bert.parameters():
            param.requires_grad = encoder_trained 
            
        self.num_labels = config.num_labels

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.classifier = nn.Linear(config.hidden_size+10, self.num_labels)
        
        self.relative_pos_embedding = nn.Embedding(4, 10)
        
        self.init_weights()
        
    def mask_pooling(self, x: Tensor, attention_mask=None):
        if attention_mask is None:
            return torch.mean(x, dim=1)
        return torch.sum(x * attention_mask.unsqueeze(2), dim=1) / torch.sum(attention_mask, dim=1, keepdim=True)

    def sequence_pooling(self, sequence_feature, attention_mask):
        if self.pooling == 'first_last_avg':
            sequence_feature = sequence_feature[-1] + sequence_feature[1]
        elif self.pooling == 'last_avg':
            sequence_feature = sequence_feature[-1]
        elif self.pooling == 'last_2_avg':
            sequence_feature = sequence_feature[-1] + sequence_feature[-2]
        elif self.pooling == 'cls':
            return sequence_feature[-1][:, 0, :]
        else:
            raise Exception("unknown pooling {}".format(self.pooling))

        return self.mask_pooling(sequence_feature, attention_mask)

    def get_encoder_feature(self, encoder_output, attention_mask):
        if self.task == 'SequenceLevel':
            return self.sequence_pooling(encoder_output, attention_mask)
        elif self.task == 'TokenLevel':
            return encoder_output[-1]
        else:
            return encoder_output[-1][:, 0, :]

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        speaker_ids=None,
        **kwargs
    ):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            return_dict=True, 
                            output_hidden_states=True
                           ).hidden_states

#         encoder_feature = self.get_encoder_feature(outputs, attention_mask)
        
        speaker_feature = self.relative_pos_embedding(speaker_ids)
#         encoder_feature = outputs[-1] + speaker_feature
                
        encoder_feature = torch.cat([outputs[-1], speaker_feature], dim=-1)
        encoder_feature = self.mask_pooling(encoder_feature, attention_mask) 
        
        encoder_feature = self.dropout(encoder_feature)
        out = self.classifier(encoder_feature)

        return out
    
from transformers import BertConfig

bert_config = BertConfig.from_pretrained('../pretrained_model/mc_bert', 
                                         num_labels=len(tm_dataset.cat2id))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

import tqdm
from tqdm import tqdm
import sklearn.metrics as sklearn_metrics
from collections import Counter


class TMPredictor(object):
    def __init__(
        self, 
        modules, 
        tokernizer, 
        cat2id
    ):

        self.modules = modules
        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.modules[0].parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
        self, 
        text_a,
        text_b
    ):
        input_ids = self.tokenizer.sequence_to_ids(text_a, text_b)  
        input_ids, input_mask, segment_ids, speaker_ids, e1_mask = input_ids

        features = {
                'input_ids': input_ids, 
                'attention_mask': input_mask, 
                'token_type_ids': segment_ids,
                'speaker_ids': speaker_ids
            }
        return features

    def _convert_to_vanilla_ids(
        self, 
        text_a, 
        text_b
    ):
        input_ids = self.tokenizer.sequence_to_ids(text_a, text_b)   

        features = {
                'input_ids': input_ids
            }
        return features

    def _get_input_ids(
        self, 
        text_a,
        text_b
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text_a, text_b)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text_a, text_b)
        elif self.tokenizer.tokenizer_type == 'customized':
            features = self._convert_to_customized_ids(text_a, text_b)
        else:
            raise ValueError("The tokenizer type does not exist") 

    def _get_module_one_sample_inputs(
        self, 
        features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
        self, 
        text,
        topk=None,
        return_label_name=True,
        return_proba=False
    ):
        if topk == None:
            topk = len(self.cat2id) if len(self.cat2id) >2 else 1
        text_a, text_b = text
        features = self._get_input_ids(text_a, text_b)
        # self.module.eval()
        
        preds = []
        probas = []
        vote_label_idx = []

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            
            logits = 0
            weight_sum = 0
            for idx, module in enumerate(self.modules):
                logit = self.modules[idx](**inputs) * 1
                logit = torch.nn.functional.softmax(logit, dim=1)

                probs, indices = logit.topk(topk, dim=1, sorted=True)
                
                preds.append(indices.cpu().numpy()[0][0])
                rank = indices.cpu().numpy()[0]
                rank_dict = {_index: _index for _index, _index in enumerate(rank)}
                probas.append([rank_dict[_index] for _index in range(len(rank))])
                
        most_ = Counter(preds).most_common(35)
#         print(most_)

        max_vote_num = most_[0][1]
        most_ = [m for m in most_ if m[1] != 1]  # 剔除1票的相同者
        most_ = [m for m in most_ if m[1] == max_vote_num]  # 只选择等于投票最大值的
        if len(most_) == 0:  # 如果全是1票
            vote_label_idx.append(Counter(preds).most_common(1)[0][0])
        elif len(most_) == 1:
            vote_label_idx.append(most_[0][0])
        else:
            prob_list_np = np.array(probas)
            select_rank = 10000
            select_m = 10000
            for m, num in most_:
                # 拿概率第m列（所有模型对第m列的概率）求和
                prob_m = prob_list_np[:, m]
                if sum(prob_m) < select_rank:
                    select_m = m
                    select_rank = sum(prob_m)

            vote_label_idx.append(select_m)

#         preds = []
#         probas = []
#         for pred_, proba_ in zip(indices.cpu().numpy()[0], probs.cpu().numpy()[0].tolist()):

#             if return_label_name:
#                 pred_ = self.id2cat[pred_]

#             preds.append(pred_)

#             if return_proba:
#                 probas.append(proba_)

#         if return_proba:
#             return list(zip(preds, probas))

        if vote_label_idx[0] == -1:
            print(most_)
            
            print(probas)

        return self.id2cat[vote_label_idx[0]]

def prob_avg_rank_in_list(prob, prob_list_np):  # 求一个数在二维数组每行的排名，然后求均值
    rank_list = []

    for i, element in enumerate(prob_list_np):
        rank = 0
        for p in element:
            if prob[i] < p:  # 概率大的放前面
                rank += 1
        rank_list.append(rank)

    return np.array(rank_list).mean()

ensemble_dl_modules = []
for file_name_ in os.listdir('../checkpoint/mcbert/'):
    if file_name_.startswith('.'):
        continue
        
    ensemble_dl_module = Bert(config=bert_config)

    ensemble_dl_module.load_state_dict(torch.load('../checkpoint/mcbert/' + file_name_))

    ensemble_dl_module.eval()
    ensemble_dl_module.to('cuda:0')
        
    ensemble_dl_modules.append(ensemble_dl_module)
    
tm_predictor_instance = TMPredictor(ensemble_dl_modules, tokenizer, tm_dataset.cat2id)

from tqdm import tqdm

submit_result = []

with codecs.open('../data/source_datasets/testb.txt', mode='r', encoding='utf8') as f:
    reader = f.readlines(f)    

data_list = []

for dialogue_ in tqdm(reader):
    dialogue_ = json.loads(dialogue_)
    for content_idx_, contents_ in enumerate(dialogue_['dialog_info']):

        terms_ = contents_['ner']

        if len(terms_) != 0:
            idx_ = 0
            for _ner_idx, term_ in enumerate(terms_):

                entity_ = dict()

                entity_['dialogue'] = dialogue_
                
                _text = dialogue_['dialog_info'][content_idx_]['text']
                _text_list = list(_text)
                _text_list.insert(term_['range'][0], '[unused1]')
                _text_list.insert(term_['range'][1]+1, '[unused2]')
                _text = ''.join(_text_list)

                if content_idx_ - 1 >= 0 and len(dialogue_['dialog_info'][content_idx_-1]) < 40:
                    forward_text = dialogue_['dialog_info'][content_idx_-1]['sender'] + ':' + dialogue_['dialog_info'][content_idx_-1]['text'] + ';'
                else:
                    forward_text = ''

                if contents_['sender'] == '医生':

                    if content_idx_ + 1 >= len(dialogue_['dialog_info']):
                        entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text
                    else:
                        entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text + ';'
                        temp_index = copy.deepcopy(content_idx_) + 1

                        speaker_flag = False
                        sen_counter = 0
                        while True:

                            if dialogue_['dialog_info'][temp_index]['sender'] == '患者':
                                sen_counter += 1
                                speaker_flag = True
                                entity_['text_a'] += dialogue_['dialog_info'][temp_index]['sender'] + ':' + dialogue_['dialog_info'][temp_index]['text'] + ';'

                            if sen_counter > 3:
                                break

                            temp_index += 1
                            if temp_index >= len(dialogue_['dialog_info']):
                                break

                elif contents_['sender'] == '患者':
                    if content_idx_ + 1 >= len(dialogue_['dialog_info']):
                        entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text
                    else:
                        entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text + ';'
                        temp_index = copy.deepcopy(content_idx_) + 1

                        speaker_flag = False
                        sen_counter = 0
                        while True:

                            sen_counter += 1
                            speaker_flag = True
                            entity_['text_a'] += dialogue_['dialog_info'][temp_index]['sender'] + ':' + dialogue_['dialog_info'][temp_index]['text'] + ';'

                            if sen_counter > 3:
                                break

                            temp_index += 1
                            if temp_index >= len(dialogue_['dialog_info']):
                                break
                else:
                    entity_['text_a'] = forward_text + dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text
                        
                    
                if term_['name'] == 'undefined':
                    add_text = '|没有标准化'
                else:
                    add_text = '|标准化为' + term_['name']

                entity_['text_b'] = term_['mention']  + add_text
                entity_['start_idx'] = term_['range'][0]
                entity_['end_idx'] = term_['range'][1] - 1

                entity_['label'] = term_['attr']
                idx_ += 1

                dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] = tm_predictor_instance.predict_one_sample([entity_['text_a'], entity_['text_b']])
    submit_result.append(dialogue_)
    
with open('./mcbert.txt', 'w') as output_data:
    for json_content in submit_result:
        output_data.write(json.dumps(json_content, ensure_ascii=False) + '\n')

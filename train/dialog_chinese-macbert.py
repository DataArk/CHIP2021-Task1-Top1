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
                    _text_list.insert(term_['range'][0], '<')
                    _text_list.insert(term_['range'][1]+1, '>')
                    _text = ''.join(_text_list)
                    
                    if contents_['sender'] == '医生':
                        
                        if content_idx_ + 1 >= len(dialogue_['dialog_info']):
                            entity_['text_a'] = dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text
                        else:
                            entity_['text_a'] = dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text + ';'
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
                            entity_['text_a'] = dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text
                        else:
                            entity_['text_a'] = dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text + ';'
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
                        entity_['text_a'] = dialogue_['dialog_info'][content_idx_]['sender'] + ':' + _text
                        
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

data_df = get_task_data('../data/source_datasets/train.jsonl')

tm_dataset = TMDataset(data_df)

def random_split_train_and_dev(data_df, split_rate=0.9):
    data_df = data_df.sample(frac=1, random_state=42)
    train_size = int(0.9 * len(data_df))
    train_df = data_df[:train_size]
    dev_df = data_df[train_size:]
    
    return train_df, dev_df

train_data_df, dev_data_df = random_split_train_and_dev(data_df)

tm_train_dataset = TMDataset(train_data_df)
tm_dev_dataset = TMDataset(dev_data_df, categories = tm_train_dataset.categories)

import transformers 
from transformers import AutoTokenizer

bert_vocab = transformers.AutoTokenizer.from_pretrained('../pretrained_model/dialog_chinese-macbert-large/')

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
        entity_end_idx = sequence_a.index('>')
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
            e11_p = sequence.index("<") + 1
            e12_p = sequence.index(">") - 1
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
        
        if len(sequence) > 150:
            print('sequence', raw_sequence_a)
        if len(sequence_mask) > 150:
            print(len(sequence_mask))
            print(len(sequence))
            print('sequence_mask', raw_sequence_a)
        if len(segment_ids) > 150:
            print('segment_ids', raw_sequence_a)
        if len(speaker_ids) > 150:
            print('speaker_ids', raw_sequence_a)
        if len(e1_mask) > 150:
            print('e1_mask', raw_sequence_a)            
            

        if return_sequence_length:
            return (sequence, sequence_mask, segment_ids, speaker_ids, e1_mask, sequence_length)

        return (sequence, sequence_mask, segment_ids, speaker_ids, e1_mask)
    
bert_vocab.add_special_tokens({'additional_special_tokens':["<", ">", "|"]})

max_seq_length=150

tokenizer = TransfomerTokenizer(bert_vocab, max_seq_length)

tm_dataset.convert_to_ids(tokenizer)

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
    

from ark_nlp.model.tc.bert import Task
from ark_nlp.factory.loss_function.focal_loss import FocalLoss
from ark_nlp.factory.utils.attack import FGM
import time
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sklearn.metrics as sklearn_metrics

from tqdm import tqdm
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from ark_nlp.factory.loss_function import get_loss
from ark_nlp.factory.optimizer import get_optimizer

class AttackTask(Task):
    
    def _on_train_begin(
        self, 
        train_data, 
        validation_data, 
        batch_size,
        lr, 
        params, 
        shuffle,
        train_to_device_cols=None,
        **kwargs
    ):
        
        if self.class_num == None:
            self.class_num = train_data.class_num  
        
        if train_to_device_cols == None:
            self.train_to_device_cols = train_data.to_device_cols
        else:
            self.train_to_device_cols = train_to_device_cols

        train_generator = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=self._collate_fn)
        self.train_generator_lenth = len(train_generator)
            
        self.optimizer = get_optimizer(self.optimizer, self.module, lr, params)
        self.optimizer.zero_grad()
        
        self.module.train()
        
        self.fgm = FGM(self.module)
        
        self._on_train_begin_record(**kwargs)
        
        return train_generator
    
    def _on_backward(
        self, 
        inputs, 
        logits, 
        loss, 
        gradient_accumulation_steps=1,
        grad_clip=None,
        **kwargs
    ):
                
        # 如果GPU数量大于1
        if self.n_gpu > 1:
            loss = loss.mean()
        # 如果使用了梯度累积，除以累积的轮数
        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
            
        loss.backward() 
        
        self.fgm.attack()
        logits = self.module(**inputs)
        attck_loss = self._get_train_loss(inputs, logits, **kwargs)
        attck_loss.backward()
        self.fgm.restore() 
        
        if grad_clip != None:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), grad_clip)
        
        self._on_backward_record(**kwargs)
        
        return loss
    
    
    
import gc
from transformers import BertConfig
from sklearn.model_selection import KFold

kf = KFold(10, shuffle=True, random_state=42)

examples = copy.deepcopy(tm_dataset.dataset)

for fold_, (train_ids, dev_ids) in enumerate(kf.split(examples)):

    tm_train_dataset.dataset = [examples[_idx] for _idx in train_ids]
    tm_dev_dataset.dataset = [examples[_idx] for _idx in dev_ids]

    bert_config = BertConfig.from_pretrained('../pretrained_model/dialog_chinese-macbert-large/', 
                                             num_labels=len(tm_train_dataset.cat2id))

    dl_module = Bert.from_pretrained('../pretrained_model/dialog_chinese-macbert-large/', 
                                            config=bert_config)

    param_optimizer = list(dl_module.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]    

    model = AttackTask(dl_module, 'adamw', 'lsce', cuda_device=0, ema_decay=0.995)
    
    model.fit(tm_train_dataset, 
              tm_dev_dataset,
              lr=2e-5,
              epochs=2, 
              batch_size=32,
              params=optimizer_grouped_parameters,
              evaluate_save=True,
              save_module_path='../checkpoint/dialog_chinese-macbert2/' + str(fold_) + '.pth'
             )
    
    del dl_module
    del model
    
    gc.collect()
    
    torch.cuda.empty_cache()
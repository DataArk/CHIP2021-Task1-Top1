import codecs
import json
from tqdm import tqdm
import copy


submit_result2 = []

with codecs.open('dialog_chinese-macbert.txt', mode='r', encoding='utf8') as f:
    reader = f.readlines(f)    

data_list = []

for dialogue_idx_, dialogue_ in enumerate(tqdm(reader)):
    dialogue_ = json.loads(dialogue_)
    submit_result2.append(dialogue_)
    
    
submit_result4 = []

with codecs.open('macbert2-f-f.txt', mode='r', encoding='utf8') as f:
    reader = f.readlines(f)    

data_list = []

for dialogue_idx_, dialogue_ in enumerate(tqdm(reader)):
    dialogue_ = json.loads(dialogue_)
    submit_result4.append(dialogue_)
    
    
submit_result3 = []

with codecs.open('macbert2-f.txt', mode='r', encoding='utf8') as f:
    reader = f.readlines(f)    

data_list = []

for dialogue_idx_, dialogue_ in enumerate(tqdm(reader)):
    dialogue_ = json.loads(dialogue_)
    submit_result3.append(dialogue_)
    

submit_result5 = []

with codecs.open('mcbert.txt', mode='r', encoding='utf8') as f:
    reader = f.readlines(f)    

data_list = []

for dialogue_idx_, dialogue_ in enumerate(tqdm(reader)):
    dialogue_ = json.loads(dialogue_)
    submit_result5.append(dialogue_)
    
    
submit_result6 = []

with codecs.open('medbert.txt', mode='r', encoding='utf8') as f:
    reader = f.readlines(f)    

data_list = []

for dialogue_idx_, dialogue_ in enumerate(tqdm(reader)):
    dialogue_ = json.loads(dialogue_)
    submit_result6.append(dialogue_)
    
    
submit_result = []

with codecs.open('macbert2-f.txt', mode='r', encoding='utf8') as f:
    reader = f.readlines(f)    

data_list = []

for dialogue_idx_, dialogue_ in enumerate(tqdm(reader)):
    dialogue_ = json.loads(dialogue_)
    for content_idx_, contents_ in enumerate(dialogue_['dialog_info']):

        terms_ = contents_['ner']

        if len(terms_) != 0:
            idx_ = 0
            for _ner_idx, term_ in enumerate(terms_):
                
                if dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] == '阳性' and dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] != submit_result3[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr']:
                        dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] = submit_result3[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr']                        
                                    
                elif dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] == '阴性' and dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] != submit_result3[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr']:                    
                            dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] = submit_result3[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr']          
                            
                elif dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] != submit_result2[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr']:
                    if submit_result2[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] == '不标注':                        
                        dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] = submit_result2[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr']
                    elif dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] == '阳性':
                        if submit_result2[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] == '其他':
                            dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] = submit_result2[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr']
                            
                elif dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] != submit_result4[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr']:
                    if dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] == '阴性':
                        if submit_result4[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] == '不标注':
                            dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] = submit_result4[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr']
                            
                elif dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] != submit_result5[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr']:
                    if dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] == '阴性':                        
                        if submit_result5[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] == '不标注':
                            dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] = submit_result5[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr']
#                         elif submit_result5[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] == '其他':
#                             dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] = submit_result5[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr']
                                
                elif dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] != submit_result6[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr']:
                    if dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] == '阳性':
                        if submit_result6[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] == '其他':
                            dialogue_['dialog_info'][content_idx_]['ner'][_ner_idx]['attr'] = submit_result6[dialogue_idx_]['dialog_info'][content_idx_]['ner'][_ner_idx]['attr']
    
    submit_result.append(dialogue_)
    
    
with open('./result.txt', 'w', encoding='utf-8') as output_data:
    for json_content in submit_result:
        output_data.write(json.dumps(json_content, ensure_ascii=False) + '\n')
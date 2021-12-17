import warnings
import pandas as pd
from transformers import (AutoModelForMaskedLM,
                          AutoTokenizer, LineByLineTextDataset,
                          DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)

warnings.filterwarnings('ignore')

def get_task_data(data_path):
    with codecs.open(data_path, mode='r', encoding='utf8') as f:
        reader = f.readlines(f)    
        
    data_list = []

    for dialogue_ in reader:
        dialogue_content = []
        dialogue_ = json.loads(dialogue_)
        
        _dialog_id = dialogue_['dialog_id']
        
        for content_idx_, contents_ in enumerate(dialogue_['dialog_info']):
            dialogue_content.append(contents_['sender'] + ':' + contents_['text'])
            
        data_list.append(';'.join(dialogue_content))
    
    return pd.DataFrame(data_list, columns=['text'])

import numpy as np
import pandas as pd
import copy
import codecs
import json


train_data = get_task_data('../data/source_datasets/train.jsonl')

test_data = get_task_data('../data/source_datasets/testa.txt')

data = pd.concat([train_data, test_data])

data['text'] = data['text'].apply(lambda x: x.replace('\n', ''))

text = '\n'.join(data.text.tolist())

with open('./text.txt', 'w') as f:
    f.write(text)
    
model_name = 'hfl/chinese-macbert-large'
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained('../pretrained_model/dialog_chinese-macbert-large')

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text.txt",  # mention train text file here
    block_size=256)

valid_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="text.txt",  # mention valid text file here
    block_size=256)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

training_args = TrainingArguments(
    output_dir="./pre",  # select model path for checkpoint
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy='steps',
    save_total_limit=2,
    eval_steps=200,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    load_best_model_at_end=True,
    prediction_loss_only=True,
    report_to="none",
    save_steps=200
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset)

trainer.train()

trainer.save_model(f'../pretrained_model/dialog_chinese-macbert-large')
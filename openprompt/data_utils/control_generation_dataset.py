# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains the logic for loading data for all Conditional Generation tasks.
"""

from openprompt.data_utils.utils import InputExample
import os
import pandas as pd
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

from openprompt.utils.logging import logger
from openprompt.data_utils.data_processor import DataProcessor


def select_examples(example_file, example_num):
    if example_file.endswith('.jsonl'):
        fin_selected =  open (example_file,'r')
        text_dict = json.loads(fin_selected.read())
        text_list = [text['comment_text'].replace('\\n','') for text in text_dict]
        text_list = [text.replace('\n','') for text in text_list]
    elif example_file.endswith('.txt'):
        fin_selected =  open (example_file,'r')
        text = fin_selected.readlines()
        text_list = [t[:-1] for t in text]
    while len(text_list)<example_num:
        text_list = text_list+text_list
    #return sample(text_list,example_num)
    return text_list[:example_num]
    
'''
def add_examples(prompts, pos_file=None, neg_file=None):
    if pos_file is not None and neg_file is not None:
        pos_examples = select_examples(pos_file,len(prompts))
        neg_examples = select_examples(neg_file,len(prompts))
        control_template_list = [ control_template.replace('POS',pos_examples[i]) for i in range(len(pos_examples)) ]
        control_template_list = [ control_template_list[i].replace('NEG',neg_examples[i]) for i in range(len(control_template_list)) ]
    elif pos_file is not None:
        pos_examples = select_examples(pos_file,len(prompts))
        control_template_list = [ control_template.replace('POS',pos_examples[i]) for i in range(len(pos_examples)) ]
    elif neg_file is not None:
        neg_examples = select_examples(neg_file,len(prompts))
        control_template_list = [ control_template.replace('NEG',neg_examples[i]) for i in range(len(neg_examples)) ]
    prompts = [control_template_list[i].replace('INP',prompts[i]) for i in range(len(prompts))]
    return prompts
'''

class ToxicityProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = None
        
    def text_split(self,text):
        text_a=' '.join(text.split(' ')[:int(len(text.split(' '))/2)])
        tgt_text=' '.join(text.split(' ')[int(len(text.split(' '))/2):])
        return text_a, tgt_text

    def get_examples(self, data_path:str, line_by_line=False, add_neg_example=False, add_pos_example=False, pos_example_filepath=None, neg_example_filepath=None):
        examples = []
        if data_path.endswith('.jsonl'):
            dataset = pd.read_json(data_path, lines=True)         #prompts/nontoxic_prompts-10k.jsonl
            prompts = pd.json_normalize(dataset['prompt'])['text']
            prompts = [t.replace('\xad','') for t in prompts]
            prompts = [t+' '+t for t in prompts]
        elif data_path.endswith('.txt'):
            #jigsaw-unintended-bias-in-toxicity-classification/toxicity_05_small.txt
            with open(data_path, encoding="utf-8") as f:
                text = f.readlines()
                text = [t[:-1] for t in text]
                prompts = [t for t in text if len(t.split(' '))>5]
                prompts = [t.replace('\xad','') for t in prompts]
        else:
            raise NotImplementedError

        if add_pos_example:
            pos_examples = select_examples(example_file=pos_example_filepath, example_num=len(prompts))
        if add_neg_example:
            neg_examples = select_examples(example_file=neg_example_filepath, example_num=len(prompts))
                
        if add_pos_example and add_neg_example:
            for i, (text, pos_example, neg_example) in enumerate(zip(prompts, pos_examples, neg_examples)):
                text_a, tgt_text = self.text_split(text)
                #example = InputExample(guid=str(i),text_a=text, tgt_text=text, meta={'pos':pos_example, 'neg':neg_example})
                example = InputExample(guid=str(i),text_a=text_a , tgt_text=tgt_text, meta={'pos':pos_example, 'neg':neg_example})
                examples.append(example)

        elif add_pos_example:
            for i, (text, pos_example) in enumerate(zip(prompts,pos_examples)):
                text_a, tgt_text = self.text_split(text)
                #example = InputExample(guid=str(i),text_a=text, tgt_text=text, meta={'pos':pos_example})
                example = InputExample(guid=str(i),text_a=text_a, tgt_text=tgt_text, meta={'pos':pos_example})
                examples.append(example)

        elif add_neg_example:
            for i, (text,neg_example) in enumerate(zip(prompts,neg_examples)):
                text_a, tgt_text = self.text_split(text)
                #example = InputExample(guid=str(i),text_a=text, tgt_text=text, meta={'neg':neg_example})
                example = InputExample(guid=str(i), text_a=text_a, tgt_text=tgt_text, meta={'neg':neg_example})
                examples.append(example)

        else:
            for i,text in enumerate(prompts):
                text_a, tgt_text = self.text_split(text)
                example = InputExample(guid=str(i), text_a=text_a, tgt_text=tgt_text)
                #example = InputExample(guid=str(i), text_a=' '.join(text.split(' ')[:int(len(text.split(' '))/2)]), tgt_text=' '.join(text.split(' ')[int(len(text.split(' '))/2):]))
                #example = InputExample(guid=str(i),text_a=text, tgt_text=text)
                examples.append(example)
        
        return examples
    

PROCESSORS = {
    "toxicity": ToxicityProcessor,
    # "e2e": E2eProcessor,
    # "dart" : DartProcessor,
}

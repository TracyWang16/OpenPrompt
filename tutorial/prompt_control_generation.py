import argparse
import torch
parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='t5')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='t5-base')

parser.add_argument("--max_seq_length", type=int, default=256)
parser.add_argument("--decoder_max_length", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=5)
parser.add_argument("--freeze_plm", dest='freeze_plm', action='store_true',default=True)
parser.add_argument("--epoch", type=int, default=1)

parser.add_argument("--add_control", dest='add_control', action='store_true',default=False)
parser.add_argument("--control_template",type=str,default=None,help='Whether to add control before input text.')
parser.add_argument("--add_pos_example",dest='add_pos_example',action='store_true',default=False)
parser.add_argument("--add_neg_example",dest='add_neg_example',action='store_true',default=False)

parser.add_argument("--data_path",type=str,default=None,help='filepath for generation prompts') #"./datasets/nontoxic_prompts-10k.jsonl"
parser.add_argument("--pos_example_filepath",type=str,default=None,help='filepath for posivite examples')
parser.add_argument("--neg_example_filepath",type=str,default=None,help='filepath for negative examples')
parser.add_argument("--model_save_dir",default=None)

args = parser.parse_args()
from openprompt.data_utils.control_generation_dataset import ToxicityProcessor



# ## Construct Template
# 
# A template can be constructed from the yaml config, but it can also be constructed by directly passing arguments.
# You can load the plm related things provided by openprompt simply by calling:

# %%
from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)
# # Try more prompt!

# You can use templates other than manual template, for example the mixedtemplate is a good place to start.
# In MixedTemplate, you can use {"soft"} to denote a tunable template. 



# Or use a mix template
from openprompt.prompts import SoftTemplate
from openprompt.prompts import MixedTemplate

# mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"special": "<eos>"} {"mask"}',num_tokens=100)

# mytemplate = SoftTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"placeholder":"text_b"} {"soft"} {"mask"}.')

if args.add_control:
    if args.add_pos_example and args.add_neg_example:
        mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft": "similar to:"} {"soft": "|", "soft_id": 1} {"meta":"pos"} {"soft_id": 1} {"soft": ", different from:"} {"soft_id": 1} {"meta":"neg"} {"soft_id": 1} {"soft":"continue from: "} {"placeholder":"text_a"} {"mask"}.')
    elif args.add_pos_example:
        mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft": "similar to:"} {"soft": "|", "soft_id": 1} {"meta":"pos"} {"soft_id": 1} {"soft":"continue from: "} {"placeholder":"text_a"} {"mask"}.')
    elif args.add_neg_example:
        mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft": "different from:"} {"soft": "|", "soft_id": 1} {"meta":"neg"} {"soft_id": 1} {"soft":"continue from: "} {"placeholder":"text_a"} {"mask"}.')
    else:
        mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft":"continue from: "} {"placeholder":"text_a"} {"mask"}.')
else:
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft":"continue from: "} {"placeholder":"text_a"} {"mask"}.')

# To better understand how does the template wrap the example, we visualize one instance.

wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])  # If you use template 3, don't worry the {"soft": "Question:"} is replace by an empty template, it is used to initialize the mixed template and then removed. 
print(wrapped_example)


# We provide a `PromptDataLoader` class to help you do all the above matters and wrap them into an `torch.DataLoader` style iterator.


from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_length, decoder_max_length=args.decoder_max_length, 
    batch_size=args.batch_size,shuffle=True, teacher_forcing=True, predict_eos_token=True,
    truncate_method="head")

# ## Now is time to build your prompt model!
# In this section we introduce using prompt to do classification, for other kinds of format, please see
# `generation_tutorial.ipynb`, `probing_tutorial.ipynb`.

print(next(iter(train_dataloader)))
# exit()

from openprompt import PromptForGeneration

use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=True,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()

from transformers import AdamW


# Using different optimizer for prompt parameters and model parameters
optimizer_grouped_parameters = [
    {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]

# optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

from transformers.optimization import get_linear_schedule_with_warmup

tot_step  = len(train_dataloader)*args.epoch
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)
# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.
from openprompt.utils.metrics import generation_metric
# Define evaluate function 

'''
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    print("test_score", score, flush=True)
'''

generation_arguments = {
    "max_length": 512,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    "bad_words_ids": None
}

global_step = 0 
tot_loss = 0 
log_loss = 0
epoch_loss = 0
epoch_log_loss = 0
# training and generation.
tot_loss = 0 
min_epoch_loss = 0
best_ckpt_dir = ''
for epoch in range(args.epoch):
    for step, inputs in enumerate(train_dataloader):
        global_step +=1
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)
        loss.backward()

        tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if global_step %500 ==0: 
            print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
            log_loss = tot_loss
    

    if epoch == 0:
        epoch_loss = tot_loss
        epoch_log_loss = tot_loss
        min_epoch_loss = tot_loss
    else:
        epoch_loss = tot_loss - epoch_log_loss
        epoch_log_loss = tot_loss

    model_save_dir = os.path.join(args.model_save_dir,'epoch_'+str(epoch))
    if min_epoch_loss > epoch_loss:
        assert not os.path.exists(model_save_dir)
        os.makedirs(model_save_dir)
        torch.save(prompt_model, os.path.join(model_save_dir,'pytorch_model.bin'))
        best_ckpt_dir = model_save_dir
        '''
        with open ('prompt_model.txt','w') as f_1:
            for para in prompt_model.parameters():
                f_1.write(str(para))
                f_1.write('\n')
        f_1.close()
        '''
        min_epoch_loss = epoch_loss

prompt_model_best = torch.load(os.path.join(best_ckpt_dir,'pytorch_model.bin'))
if use_cuda:
    prompt_model_best =  prompt_model_best.cuda()
prompt_model_best.eval()
#evaluate(prompt_model_best, test_dataloader)


# %%


args = parser.parse_args()

from openprompt.data_utils.control_generation_dataset import ToxicityProcessor

dataset = {}
dataset['train'] = ToxicityProcessor().get_examples(data_path=args.data_path, add_neg_example=args.add_neg_example, add_pos_example=aegs.add_pos_example, neg_example_filepath=args.neg_example_filepath,pos_example_filepath=args.pos_example_filepath)

# ## Construct Template
# 
# A template can be constructed from the yaml config, but it can also be constructed by directly passing arguments.
# You can load the plm related things provided by openprompt simply by calling:

# %%
from openprompt.plms import load_plm

plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)

from openprompt.prompts import MixedTemplate
if args.add_control:
    if args.add_pos_example and args.add_neg_example:
        mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft": "similar to:"} {"soft": "|", "soft_id": 1} {"meta":"pos"} {"soft_id": 1} {"soft": ", different from:"} {"soft_id": 1} {"meta":"neg"} {"soft_id": 1} {"soft":"continue from: "} {"placeholder":"text_a"} {"mask"}.')
    elif args.add_pos_example:
        mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft": "similar to:"} {"soft": "|", "soft_id": 1} {"meta":"pos"} {"soft_id": 1} {"soft":"continue from: "} {"placeholder":"text_a"} {"mask"}.')
    elif args.add_neg_example:
        mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft": "different from:"} {"soft": "|", "soft_id": 1} {"meta":"neg"} {"soft_id": 1} {"soft":"continue from: "} {"placeholder":"text_a"} {"mask"}.')
else:
        mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft":"continue from: "} {"placeholder":"text_a"} {"mask"}.')

wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])  # If you use template 3, don't worry the {"soft": "Question:"} is replace by an empty template, it is used to initialize the mixed template and then removed. 
print(wrapped_example)

from openprompt import PromptDataLoader

train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
    tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_length, decoder_max_length=args.decoder_max_length, 
    batch_size=args.batch_size,shuffle=True, teacher_forcing=True, predict_eos_token=True,
    truncate_method="head")


from openprompt import PromptForGeneration
use_cuda = True
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=args.freeze_plm,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()

from transformers import AdamW

# Using different optimizer for prompt parameters and model parameters
optimizer_grouped_parameters = [
    {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]

# optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)

from transformers.optimization import get_linear_schedule_with_warmup

tot_step  = len(train_dataloader)*args.epoch
scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)
# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.

'''
from openprompt.utils.metrics import generation_metric
# Define evaluate function 
def evaluate(prompt_model, dataloader):
    generated_sentence = []
    groundtruth_sentence = []

    for step, inputs in enumerate(dataloader):
        if use_cuda:
            inputs = inputs.cuda()
        _, output_sentence = prompt_model.generate(inputs, **generation_arguments)
        generated_sentence.extend(output_sentence)
        groundtruth_sentence.extend(inputs['tgt_text'])
    score = generation_metric(generated_sentence, groundtruth_sentence, "sentence_bleu")
    print("test_score", score, flush=True)
'''

generation_arguments = {
    "max_length": 512,
    "max_new_tokens": None,
    "min_length": 5,
    "temperature": 1.0,
    "do_sample": False,
    "top_k": 0,
    "top_p": 0.9,
    "repetition_penalty": 1.0,
    "num_beams": 5,
    "bad_words_ids": None
}

global_step = 0 
tot_loss = 0 
log_loss = 0
epoch_loss = 0
epoch_log_loss = 0
# training and generation.
tot_loss = 0 
min_epoch_loss = 0

for epoch in range(args.epoch):
    for step, inputs in enumerate(train_dataloader):
        global_step +=1
        if use_cuda:
            inputs = inputs.cuda()
        loss = prompt_model(inputs)
        loss.backward()

        tot_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if global_step %500 ==0: 
            print("Epoch {}, global_step {} average loss: {} lr: {}".format(epoch, global_step, (tot_loss-log_loss)/500, scheduler.get_last_lr()[0]), flush=True)
            log_loss = tot_loss

    if epoch == 0:
        epoch_loss = tot_loss
        epoch_log_loss = tot_loss
        min_epoch_loss = tot_loss
    else:
        epoch_loss = tot_loss - epoch_log_loss
        epoch_log_loss = tot_loss

    model_save_dir = os.path.join(args.model_save_dir,'epoch_'+str(epoch))

    if min_epoch_loss > epoch_loss:
        assert not os.path.exists(model_save_dir)
        os.makedirs(model_save_dir)

        torch.save(prompt_model, os.path.join(model_save_dir,'pytorch_model.bin'))
        prompt_model_new = torch.load(os.path.join(model_save_dir,'pytorch_model.bin'))
        min_epoch_loss = epoch_loss

if use_cuda:
    prompt_model_new =  prompt_model_new.cuda()
prompt_model_new.eval()
#evaluate(prompt_model_new, test_dataloader)


#evaluate(prompt_model, test_dataloader)


import argparse
import torch
import os
import pandas as pd

parser = argparse.ArgumentParser("")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--adam_epsilon", type=float, default=1e-8)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
parser.add_argument("--plm_eval_mode", action="store_true")
parser.add_argument("--model", type=str, default='t5')  # tested model are gpt2/t5
parser.add_argument("--model_name_or_path", default='t5-base')
parser.add_argument("--finetuned_model_path", default='t5-base')

parser.add_argument("--max_seq_length", type=int, default=256)
parser.add_argument("--min_seq_length", type=int, default=0)
parser.add_argument("--decoder_max_length", type=int, default=256) #only for enc-dec models
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--accumulation_step", type=int, default=32) 
parser.add_argument("--freeze_plm", dest='freeze_plm', action='store_true',default=False)
parser.add_argument("--epoch", type=int, default=3)

parser.add_argument("--add_control", dest='add_control', action='store_true',default=False)
parser.add_argument("--control_template",type=str,default=None,help='Whether to add control before input text.')
parser.add_argument("--add_pos_example",dest='add_pos_example',action='store_true',default=False)
parser.add_argument("--add_neg_example",dest='add_neg_example',action='store_true',default=False)

parser.add_argument("--data_path",type=str,default=None,help='filepath for generation prompts') #"./datasets/nontoxic_prompts-10k.jsonl"
parser.add_argument("--pos_example_filepath",type=str,default=None,help='filepath for posivite examples')
parser.add_argument("--neg_example_filepath",type=str,default=None,help='filepath for negative examples')
parser.add_argument("--model_save_dir",default=None)
parser.add_argument("--generated_file_path",default=None)

parser.add_argument("--top_k", type=int, default=0)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--T", type=float, default=1.0)


args = parser.parse_args()

from openprompt.data_utils.control_generation_dataset import ToxicityProcessor
from openprompt.data_utils.conditional_generation_dataset import WebNLGProcessor

if not args.plm_eval_mode:
    assert not os.path.exists(args.model_save_dir)

dataset = {}
if not args.plm_eval_mode:
    dataset['train'] = ToxicityProcessor().get_examples(data_path=args.data_path, add_neg_example=args.add_neg_example, add_pos_example=args.add_pos_example, neg_example_filepath=args.neg_example_filepath,pos_example_filepath=args.pos_example_filepath)
else:
    dataset['test'] = ToxicityProcessor().get_examples(data_path=args.data_path, add_neg_example=args.add_neg_example, add_pos_example=args.add_pos_example, neg_example_filepath=args.neg_example_filepath,pos_example_filepath=args.pos_example_filepath)

# ## Construct Template
# 
# A template can be constructed from the yaml config, but it can also be constructed by directly passing arguments.
# You can load the plm related things provided by openprompt simply by calling:

# %%
from openprompt.plms import load_plm
plm, tokenizer, model_config, WrapperClass = load_plm(args.model, args.model_name_or_path)
#if args.plm_eval_mode:
#    model_for_generate = torch.load(os.path.join(args.finetuned_model_path,'pytorch_model.bin'))
#dataset['train'] = dataset['train'][0:40000]
from openprompt.prompts import MixedTemplate
from openprompt.prompts import SoftTemplate

if args.add_control:
    if args.add_pos_example and args.add_neg_example:
        #mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft": "similar to:"} {"soft": "|", "soft_id": 1} {"meta":"pos"} {"soft_id": 1} {"soft": ", different from:"} {"soft_id": 1} {"meta":"neg"} {"soft_id": 1} {"soft":"continue from: "} {"placeholder":"text_a"} {"mask"}.')
        #mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft": "similar to:"} {"soft": "|", "soft_id": 1} {"meta":"pos","shortable":True} {"soft_id": 1} {"soft": ", different from:"} {"soft_id": 1} {"meta":"neg","shortable":True} {"soft_id": 1} {"mask"}.')
        mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft": "similar to:"} {"meta":"pos","shortable":True} {"soft": ", different from:"} {"soft_id": 1} {"meta":"neg","shortable":True} {"soft_id": 1} {"placeholder":"text_a"} {"mask"}.')
    elif args.add_pos_example:
        #mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft": "similar to:"} {"soft": "|", "soft_id": 1} {"meta":"pos"} {"soft_id": 1} {"soft":"continue from: "} {"placeholder":"text_a"} {"mask"}.')
        #mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft": "similar to:"} {"soft": "|", "soft_id": 1} {"meta":"pos","shortable":True} {"soft_id": 1} {"mask"}.')
        mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft": "similar to:"} {"soft": "|"} {"meta":"pos","shortable":True} {"soft": "|"} {"placeholder":"text_a"} {"mask"}.')
    elif args.add_neg_example:
        #mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft": "different from:"} {"soft": "|", "soft_id": 1} {"meta":"neg"} {"soft_id": 1} {"soft":"continue from: "} {"placeholder":"text_a"} {"mask"}.')
        mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft": "different from:"} {"soft": "|", "soft_id": 1} {"meta":"neg","shortable":True} {"soft_id": 1} {"placeholder":"text_a"} {"mask"}.')
    else:
        #mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft":"continue from: "} {"placeholder":"text_a"} {"mask"}.')
        mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence: "} {"placeholder":"text_a"} {"mask"}.')
        #SoftTemplate(model=plm, tokenizer=tokenizer, text='{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"placeholder":"text_b"} {"soft"} {"mask"}.')
else:
    #mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"soft":"Write a sentence"} {"soft":"continue from: "} {"placeholder":"text_a"} {"mask"}.')
    mytemplate = MixedTemplate(model=plm, tokenizer=tokenizer, text='{"mask"}.')

#wrapped_example = mytemplate.wrap_one_example(dataset['train'][0])  # If you use template 3, don't worry the {"soft": "Question:"} is replace by an empty template, it is used to initialize the mixed template and then removed. 
#print(wrapped_example)

from openprompt import PromptDataLoader

if not args.plm_eval_mode:
    '''
    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_length, decoder_max_length=args.decoder_max_length, 
        batch_size=args.batch_size,shuffle=False, teacher_forcing=True, predict_eos_token=True,
        truncate_method="head")
    '''
    train_dataloader = PromptDataLoader(dataset=dataset["train"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_length, decoder_max_length=args.decoder_max_length, 
        batch_size=args.batch_size,shuffle=True, teacher_forcing=True, predict_eos_token=True,
        truncate_method="head")
else:
    test_dataloader = PromptDataLoader(dataset=dataset["test"], template=mytemplate, tokenizer=tokenizer, 
        tokenizer_wrapper_class=WrapperClass, max_seq_length=args.max_seq_length, decoder_max_length=args.decoder_max_length, 
        batch_size=args.batch_size,shuffle=True, teacher_forcing=True, predict_eos_token=True,
        truncate_method="head")    

from openprompt import PromptForGeneration
use_cuda = True
#prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=args.freeze_plm,tokenizer=tokenizer, plm_eval_mode=args.plm_eval_mode)
prompt_model = PromptForGeneration(plm=plm,template=mytemplate, freeze_plm=False, plm_eval_mode=args.plm_eval_mode)
if use_cuda:
    prompt_model=  prompt_model.cuda()
'''
for n,p in prompt_model.template.named_parameters():
    print(n)
# Using different optimizer for prompt parameters and model parameters
optimizer_grouped_parameters = [
    {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]

# optimizer1 = AdamW(optimizer_grouped_parameters1, lr=1e-4)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
'''
from transformers import AdamW

no_decay = ['bias', 'LayerNorm.weight']

# it's always good practice to set no decay to biase and LayerNorm parameters
optimizer_grouped_parameters1 = [
    {'params': [p for n, p in prompt_model.plm.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    {'params': [p for n, p in prompt_model.plm.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# Using different optimizer for prompt parameters and model parameters
optimizer_grouped_parameters2 = [
    {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
]

optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.lr, eps=args.adam_epsilon)
optimizer2 = AdamW(optimizer_grouped_parameters2, lr=args.lr, eps=args.adam_epsilon)

from transformers.optimization import get_linear_schedule_with_warmup

if not args.plm_eval_mode:
    tot_step  = int( len(train_dataloader)*args.epoch/args.accumulation_step )
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, 0, tot_step)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, 0, tot_step)

# We provide generation a generation metric, you can also define your own. Note that it's not directly comparable to WebNLG's scripts evaluation.

generation_arguments = {
    "max_length": args.max_seq_length,
    "max_new_tokens": None,
    "min_length": args.min_seq_length,
    "temperature": args.T,
    "do_sample": False,
    "top_k": args.top_k,
    "top_p": args.top_p,
    "repetition_penalty": 1.0, #repetition_penalty != 1.0:
    "num_beams": 1,
    "bad_words_ids": None
}

global_step = 0 
tot_loss = 0 
log_loss = 0
epoch_loss = 0
epoch_log_loss = 0
best_epoch = 0
# training and generation.
tot_loss = 0 
min_epoch_loss = 0
true_steps = 0
'''
if not os.path.exists(os.path.join(args.model_save_dir,'epoches')):
    os.makedirs(os.path.join(args.model_save_dir,'epoches'))
'''
if not args.plm_eval_mode:
    if not os.path.exists(os.path.join(args.model_save_dir,'epoches')):
        os.makedirs(os.path.join(args.model_save_dir,'epoches'))
    for epoch in range(1,args.epoch+1):
        for step, inputs in enumerate(train_dataloader):
            global_step +=1
            if use_cuda:
                inputs = inputs.cuda()
            loss = prompt_model(inputs)

            #参考https://github.com/TracyWang16/GeDi/blob/master/train_GeDi.py
            #accumulation 
            loss = loss / args.accumulation_step
            loss.backward()
            tot_loss += loss.item()

            if (step+1) % args.accumulation_step == 0 :
                true_steps = true_steps + 1
                torch.nn.utils.clip_grad_norm_(mytemplate.parameters(), args.max_grad_norm)
                optimizer1.step()
                optimizer2.step()
                scheduler1.step()
                scheduler2.step()
                optimizer1.zero_grad()
                optimizer2.zero_grad()

            if true_steps %50 ==0: 
                print("Epoch {}, true_step {} global_step {} average loss: {} lr: {}".format(epoch, true_steps, global_step, (tot_loss-log_loss)/50*args.accumulation_step, scheduler1.get_last_lr()[0]), flush=True)
                #if not os.path.exists(os.path.join(args.model_save_dir,'log.txt')):
                #    os.makedirs(os.path.join(args.model_save_dir,'log.txt'))
                with open(os.path.join(args.model_save_dir,'log.txt'), 'a') as f_log:
                    f_log.write("Epoch {}, true_step {} global_step {} average loss: {} lr: {} \n".format(epoch, true_steps, global_step, (tot_loss-log_loss)/50*args.accumulation_step, scheduler1.get_last_lr()[0]))
                    f_log.close()
                log_loss = tot_loss

        if epoch == 0:
            epoch_loss = tot_loss
            epoch_log_loss = tot_loss
            min_epoch_loss = tot_loss
        else:
            epoch_loss = tot_loss - epoch_log_loss
            epoch_log_loss = tot_loss

        print("Epoch {}, loss:{} \n".format(epoch,epoch_loss))
        with open(os.path.join(args.model_save_dir,'log.txt'), 'a') as f_log:
            f_log.write("Epoch {}, loss:{} \n".format(epoch,epoch_loss))
            f_log.close()

        if epoch_loss<=min_epoch_loss:    
            min_epoch_loss = epoch_loss
            best_epoch = epoch

        #if not os.path.exists(os.path.join(args.model_save_dir,'epoches')):
        #os.makedirs(os.path.join(args.model_save_dir,'epoches'))
        if epoch % 2==0:
            model_save_dir = os.path.join(args.model_save_dir,'epoches','epoch_'+str(epoch))
            assert not os.path.exists(model_save_dir)
            os.makedirs(model_save_dir)

            torch.save(prompt_model, os.path.join(model_save_dir,'pytorch_model.bin'))
        #prompt_model_new = torch.load(os.path.join(model_save_dir,'pytorch_model.bin'))

    print("Best Epoch {} \n".format(best_epoch))
    with open(os.path.join(args.model_save_dir,'log.txt'), 'a') as f_log:
        f_log.write("Best Epoch {} \n".format(best_epoch))
        f_log.close()
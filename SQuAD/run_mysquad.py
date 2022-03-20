# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
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
""" Finetuning the library models for question-answering on SQuAD (Bert, XLM, XLNet)."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import glob
import gc
import apex
#from apex.normalization.fused_layer_norm import FusedLayerNorm
import pickle
import collections
import sys
sys.path.append('../')
from utils import (get_answer)


import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from tensorboardX import SummaryWriter

from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer,
                                  XLMConfig, XLMForQuestionAnswering,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForQuestionAnswering,
                                  XLNetTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_squad import (read_squad_examples, convert_examples_to_features,
                         RawResult, write_predictions,
                         RawResultExtended, write_predictions_extended)

# The follwing import is the official SQuAD evaluation script (2.0).
# You can remove it from the dependencies if you are using this script outside of the library
# We've added it here for automated tests (see examples/test_examples.py file)
from utils_squad_evaluate import EVAL_OPTS, main as evaluate_on_squad

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) \
                  for conf in (BertConfig, XLNetConfig, XLMConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForQuestionAnswering, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForQuestionAnswering, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForQuestionAnswering, XLMTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def to_list(tensor):
    return tensor.detach().cpu().tolist()

class parametersClass:
    def __init__(self,first_batch,bert_model,old_output_dir,output_dir,train_file,trainFlag):
        self.first_batch = first_batch
        self.model_type = 'bert'
        self.model_name_or_path = bert_model
        self.do_train = trainFlag
        self.max_seq_length = 384
        self.max_query_length = 64
        self.doc_stride = 128
        self.train_file = train_file
        self.local_rank = -1
        self.no_cuda = False
        self.output_dir = output_dir
        self.old_output_dir = old_output_dir
        self.config_name = ""
        self.tokenizer_name = ""
        self.do_lower_case = True
        self.trainFlag = trainFlag
        self.fp16 = True
        self.seed = 42
        self.per_gpu_train_batch_size = 8
        self.per_gpu_eval_batch_size = 8
        self.learning_rate  = 5e-5
        self.num_train_epochs = 3.0
        self.verbose_logging = False
        self.max_answer_length = 30
        self.warmup_steps = 0
        self.max_grad_norm = 1.0
        self.adam_epsilon = 1e-8
        self.weight_decay = 0.0
        self.logging_steps = 200
        self.max_steps = -1
        self.gradient_accumulation_steps = 1
        self.fp16_opt_level = 'O1'
        self.save_steps = 50

        
def getFeatures(args,train_file,tokenizer):
    evaluate = not args.trainFlag
    cached_train_examples_file = args.train_file
    with open(cached_train_examples_file, "rb") as reader:
        train_examples = pickle.load(reader)
    train_features = convert_examples_to_features(examples=train_examples,
                                            tokenizer=tokenizer,
                                            max_seq_length=args.max_seq_length,
                                            doc_stride=args.doc_stride,
                                            max_query_length=args.max_query_length,
                                            is_training=not evaluate)

    if (evaluate):
        return [train_examples,train_features]
    
     # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in train_features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in train_features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask)

    return dataset
        

def train(args, train_dataset, model, tokenizer, output_dir):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
#     if args.fp16:
#         try:
#             from apex import amp
#         except ImportError:
#             raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
#         model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    #multi-gpu training (should be after apex fp16 initialization)
#     if args.n_gpu > 1:
#         model = torch.nn.DataParallel(model)

#     # Distributed training (should be after apex fp16 initialization)
#     if args.local_rank != -1:
#         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
#                                                           output_device=args.local_rank,
#                                                           find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    del train_dataset
    gc.collect()
#     torch.cuda.empty_cache()

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':       batch[0],
                      'attention_mask':  batch[1], 
                      'token_type_ids':  None if args.model_type == 'xlm' else batch[2],  
                      'start_positions': batch[3], 
                      'end_positions':   batch[4]}
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5],
                               'p_mask':    batch[6]})
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

#             if args.n_gpu > 1:
#                 loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

#             if args.fp16:
#                 with amp.scale_loss(loss, optimizer) as scaled_loss:
#                     scaled_loss.backward()
#                 torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
#             else:
#                 loss.backward()
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):                        
                        os.makedirs(output_dir)
                        
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training                    
                    model_to_save.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()
        
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    print('saving my finetuned model')
    model_to_save.save_pretrained(output_dir)
    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
    tokenizer.save_pretrained(args.output_dir)

    del model
    del tokenizer
    gc.collect()
#     torch.cuda.empty_cache() 
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Reserved CUDA memory is',r)
    print('Allocated CUDA memory is', a)
    print('Free CUDA memory is', f)    


def fineTune(first_batch_flag,bert_model,old_output_dir,output_dir,train_file):
    """ Finetuning Bert-SQuAD"""
    trainFlag = True
    evaluate = not trainFlag
    args = parametersClass(first_batch_flag,bert_model,old_output_dir,output_dir,train_file,trainFlag)
    print('Is this Traning?', args.do_train)
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda: 
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    
    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    if args.first_batch: # initialize vanilla bert for the training with the first batch
        print("Loading the initial Bert model")
        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        print(model_class)
        print(config_class)
        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    else: # else initialize from the previously trained model
        print("Loading existing model")
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.old_output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.old_output_dir) 
        model = model_class.from_pretrained(args.old_output_dir,config=config)
        
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    
    if not os.path.exists(args.output_dir):
        print('Creating output_dir')
        os.makedirs(args.output_dir)

    # Training
    if args.do_train:
        train_dataset = getFeatures(args,train_file,tokenizer)        
        train(args, train_dataset, model, tokenizer, output_dir)
        
    

def getFineTunedContext(first_batch_flag,bert_model,old_output_dir,output_dir,train_file):   
    
    trainFlag = False
    evaluate = not trainFlag
    args = parametersClass(first_batch_flag,bert_model,old_output_dir,output_dir,train_file,trainFlag)
    print('Is this Traning?', args.do_train)

    if args.local_rank == -1 or args.no_cuda:  
        device = torch.device("cuda" if torch.cuda.is_available()
                         and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        print('n_gpu',n_gpu)
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        
    if args.first_batch: # initialize vanilla bert for the training with the first batch
        print("Loading the initial Bert model")
        # Load pretrained model and tokenizer
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        args.model_type = args.model_type.lower()
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
        config.output_hidden_states = True
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
        model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)
        print(model_class)
        print(config_class)

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    else: # else initialize from the previously trained model
        print("Loading your finetuned model")
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]        
        config = config_class.from_pretrained(args.output_dir, output_hidden_states=True)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model = model_class.from_pretrained(args.output_dir, config=config)        
       
    
    [eval_examples,eval_features] = getFeatures(args,train_file,tokenizer)
      
    del tokenizer
    gc.collect()
    
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Reserved CUDA memory is',r)
    print('Allocated CUDA memory is', a)
    print('Free CUDA memory is', f)
    
    print(device)
    model.to(device)
    
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in eval_features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in eval_features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask)    
   
    gc.collect()

    eval_sampler = SequentialSampler(eval_data)
    batch_size = 1
    eval_dataloader = DataLoader(eval_data,
                         sampler=eval_sampler,
                         batch_size=batch_size) 


    indices = []
    stacked_embeddings = torch.zeros([20,768]) # Hardcoding here!
    model.eval()
    all_results = []
    counter = -1
    for batch in tqdm(eval_dataloader, desc="Evaluating"):  
        counter+=1
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': None if args.model_type == 'xlm' else batch[2]  # XLM don't use segment_ids
                     }
            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4],
                               'p_mask':    batch[5]})
            batch_outputs = model(**inputs) 
            batch_hidden_embedding_outputs = torch.mean(batch_outputs[2][0],1)
            start_index = counter*batch_size 
            end_index = start_index + batch_size 
            stacked_embeddings[start_index:end_index] = batch_hidden_embedding_outputs


    print('Shape of stacked embeddings',stacked_embeddings.shape)

    del model
    gc.collect()

    return stacked_embeddings

    

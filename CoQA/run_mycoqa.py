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
"""Run BERT on CoQA."""

from __future__ import absolute_import, division, print_function

import logging
import os
import random
import sys
from io import open
import json
import gc

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from modeling import BertForCoQA
from bert import AdamW, WarmupLinearSchedule, BertTokenizer

from run_coqa_dataset_utils import read_coqa_examples, convert_examples_to_features, RawResult, write_predictions, score
# from parallel import DataParallelCriterion, DataParallelModel, gather

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import collections

logger = logging.getLogger(__name__)

class parametersClass:
    def __init__(self,first_batch,bert_model,old_output_dir,output_dir,train_file,trainFlag):
        self.first_batch = first_batch
        self.bert_model = bert_model
        self.output_dir = output_dir
        self.old_output_dir = old_output_dir
        self.train_file = train_file
        self.max_seq_length = 512
        self.doc_stride = 128
        self.max_query_length = 64
        self.max_answer_length = 30
        self.type_model = 'bert'
        self.warmup_proportion = 0.1
        self.n_best_size=20
        self.seed = 42
        self.loss_scale=0.0
        self.do_train = trainFlag
        self.do_predict = not trainFlag
        self.train_batch_size=10
        self.predict_batch_size=10
        self.learning_rate=3e-5
        self.num_train_epochs=2.0
        self.no_cuda=False
        self.gradient_accumulation_steps=1
        self.verbose_logging = True
        self.do_lower_case=True
        self.local_rank=-1
        self.fp16=True
        self.fp16_opt_level='O1'
        self.overwrite_output_dir=True,
        self.weight_decay=0.0
        self.null_score_diff_threshold=0.0
        self.server_ip = ''
        self.server_port = ''
        self.logfile=None
        self.logmode=None
        self.tensorboard = True
        self.qa_tag = False
        self.history_len=2
        self.adam_epsilon=1e-8
        self.max_grad_norm=1.0
        self.logging_steps=50

def getFeatures(args,train_file,tokenizer):
    if args.local_rank in [-1, 0] and args.tensorboard:
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter()
    # Prepare data loader
    # try train_examples
    cached_train_examples_file = args.train_file
    with open(cached_train_examples_file, "rb") as reader:
        train_examples = pickle.load(reader)
    # try train_features

    train_features = convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
    )
    return [train_examples,train_features]

def fineTune(first_batch,bert_model,old_output_dir,output_dir,train_file):
    """ Finetuning Bert-CoQA"""
    trainFlag = True
    args = parametersClass(first_batch,bert_model,old_output_dir, output_dir,train_file,trainFlag)
    if args.server_ip and args.server_port:
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port),
                            redirect_output=True)
        ptvsd.wait_for_attach()        
        

    if args.local_rank == -1 or args.no_cuda:
        print('Cuda is available?',torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
        print('N_GPU', n_gpu)
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
        filename=args.logfile,
        filemode=args.logmode)

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".
        format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1"
            .format(args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True."
        )

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir
    ) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier(
        )  # Make sure only the first process in distributed training will download model & vocab

    if args.first_batch: # initialize vanilla bert for the training with the first batch
        print("Loading the initial Bert model")
        tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case)
        model = BertForCoQA.from_pretrained(args.bert_model)
        if args.local_rank == 0:
            torch.distributed.barrier()
    else: # else initialize from the previously trained model
        print("Loading existing model")
        model = BertForCoQA.from_pretrained(args.old_output_dir)
        tokenizer = BertTokenizer.from_pretrained(
            args.old_output_dir, do_lower_case=args.do_lower_case)
        print("\ndo_lower_case",args.do_lower_case)       
        

    model.to(device)
    print(device)


    if args.do_train:        
       
        [train_examples,train_features] = getFeatures(args,train_file,tokenizer)
        
        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        
        del train_examples
        gc.collect()
        torch.cuda.empty_cache()

        all_input_ids = torch.tensor([f.input_ids for f in train_features],
                                     dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features],
                                      dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features],
                                       dtype=torch.long)
        all_start_positions = torch.tensor(
            [f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor(
            [f.end_position for f in train_features], dtype=torch.long)
        all_rational_mask = torch.tensor(
            [f.rational_mask for f in train_features], dtype=torch.long)
        all_cls_idx = torch.tensor([f.cls_idx for f in train_features],
                                   dtype=torch.long)
        
        del train_features 
        gc.collect()
        torch.cuda.empty_cache()
        
        train_data = TensorDataset(all_input_ids, all_input_mask,
                                   all_segment_ids, all_start_positions,
                                   all_end_positions, all_rational_mask,
                                   all_cls_idx)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data,
                                      sampler=train_sampler,
                                      batch_size=args.train_batch_size)
        num_train_optimization_steps = len(
            train_dataloader
        ) // args.gradient_accumulation_steps * args.num_train_epochs

        # Prepare optimizer
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            args.weight_decay
        }, {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]

        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(args.warmup_proportion * num_train_optimization_steps), t_total=num_train_optimization_steps)

#         if args.fp16:
#             try:
#                 from apex import amp
#             except ImportError:
#                 raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
#             model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
#         if args.first_batch:
#             if args.fp16:
#                 try:
#                     from apex import amp
#                 except ImportError:
#                     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
#                 model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
#         else:
#             if args.fp16:
#                 try:
#                     from apex import amp
#                 except ImportError:
#                     raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
#             _, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        
#         if n_gpu > 1:
#             model = torch.nn.DataParallel(model)
        
#         if args.local_rank != -1:
#             model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
#                                                             output_device=args.local_rank,
#                                                             find_unused_parameters=True)

        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0


        
        model.train()
        
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            for step, batch in enumerate(
                    tqdm(train_dataloader,
                         desc="Iteration",
                         disable=args.local_rank not in [-1, 0])):
                batch = tuple(
                    t.to(device)
                    for t in batch)  # multi-gpu does scattering it-self
                input_ids, input_mask, segment_ids, start_positions, end_positions, rational_mask, cls_idx = batch
                loss = model(input_ids, segment_ids, input_mask,
                             start_positions, end_positions, rational_mask,
                             cls_idx)
                # loss = gather(loss, 0)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

#                 if args.fp16:
#                     with amp.scale_loss(loss, optimizer) as scaled_loss:
#                         scaled_loss.backward()
#                     if args.max_grad_norm > 0:
#                         torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
#                 else:
#                     loss.backward()
#                     if args.max_grad_norm > 0:
#                         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                loss.backward()
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    model.zero_grad()
                    global_step += 1
                    if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        if args.tensorboard:
                            tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                            tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                        else:
                            logger.info('Step: {}\tLearning rate: {}\tLoss: {}\t'.format(global_step, scheduler.get_lr()[0], (tr_loss - logging_loss)/args.logging_steps))
                        logging_loss = tr_loss

    if args.do_train and (args.local_rank == -1
                          or torch.distributed.get_rank() == 0):
        
        model_to_save = model.module if hasattr(model, 'module') else model 
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    

    del model
    del tokenizer
#     del scheduler
#     del tr_loss
#     del logging_loss
#     del args
#     del loss
#     del optimizer
#     del param_optimizer
#     del global_step
    gc.collect()
    torch.cuda.empty_cache() 
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0) 
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved
    print('Reserved CUDA memory is',r)
    print('Allocated CUDA memory is', a)
    print('Free CUDA memory is', f)    

        
#     for obj in gc.get_objects():
#         try:
#             if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
#                 print(type(obj), obj.size())
#         except:
#             pass



def getFineTunedContext(first_batch_flag,bert_model,old_output_dir,output_dir,train_file):   
    
    #bert_model = 'bert-base-uncased'
    trainFlag = False
    args = parametersClass(first_batch_flag,bert_model,old_output_dir,output_dir,train_file,trainFlag)
    print('Is this Traning?', args.do_train)

    if args.local_rank == -1 or args.no_cuda:
        print('Cuda is available?',torch.cuda.is_available())
        device = torch.device("cuda" if torch.cuda.is_available()
                              and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        
    if args.first_batch: # initialize vanilla bert for the training with the first batch
        print("Loading the initial Bert model")
        tokenizer = BertTokenizer.from_pretrained(
            args.bert_model, do_lower_case=args.do_lower_case)
        model = BertForCoQA.from_pretrained(args.bert_model)
        if args.local_rank == 0:
            torch.distributed.barrier()
    else: # else initialize from the previously trained model
        print("Loading your finetuned model")
        model = BertForCoQA.from_pretrained(args.output_dir)
        tokenizer = BertTokenizer.from_pretrained(
            args.output_dir, do_lower_case=args.do_lower_case) 
    
    [eval_examples,eval_features] = getFeatures(args,train_file,tokenizer)
    
    outputs = torch.zeros(len(eval_examples),args.max_seq_length,768)
    
    #del eval_examples    
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    model.to(device)
    print(device)
    
    all_input_ids = torch.tensor([f.input_ids for f in eval_features],dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features],dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features],dtype=torch.long)
    all_example_index = torch.tensor([f.example_index for f in eval_features],dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask,all_segment_ids, all_example_index)
    
    del eval_features
    gc.collect()
    torch.cuda.empty_cache()

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data,
                         sampler=eval_sampler,
                         batch_size=1)



    #all_outputs = torch.zeros(len(eval_features),max_seq_length,768)
    indices = []
    outputs_dict = collections.defaultdict(list)
    model.eval()
    counter=-1
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        with torch.no_grad():
            dummyA, dummyB, dummyC, dummyD, dummyE, batch_outputs = model(input_ids, segment_ids, input_mask)
        for i, example_index in enumerate(example_indices):
            counter+=1
            outputs_dict[example_index.item()].append(batch_outputs[0].cpu())
            indices.append(example_index)
            #all_outputs[counter]=batch_outputs[0].detach()
    print(counter+1)


    del model
    del input_ids
    del input_mask
    del segment_ids
    gc.collect()
    torch.cuda.empty_cache()
    
    for example_index,list_of_outputs in outputs_dict.items():
        num_lists = len(list_of_outputs)
        sum_array = np.zeros([args.max_seq_length,768])
        for tensor in list_of_outputs:
            numpy_array = np.array(tensor.squeeze().cpu());
            sum_array+=numpy_array;
        avg_array = sum_array/num_lists
        outputs[example_index] = torch.tensor(avg_array);
        stacked_embeddings = torch.mean(outputs,1)
        
    del batch_outputs
    del dummyA
    del dummyB
    del dummyC
    del dummyD
    del dummyE
    del outputs_dict
    del args
    gc.collect()
    torch.cuda.empty_cache()
    
#     t = torch.cuda.get_device_properties(0).total_memory
#     r = torch.cuda.memory_reserved(0) 
#     a = torch.cuda.memory_allocated(0)
#     f = r-a  # free inside reserved
#     print('Reserved CUDA memory is',r)
#     print('Allocated CUDA memory is', a)
#     print('Free CUDA memory is', f)
    

    return stacked_embeddings

    

        




# if __name__ == "__main__":
#     main()

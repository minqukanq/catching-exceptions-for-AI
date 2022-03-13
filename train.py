'''
Author : Mingu Kang
Date   : Nov 2021
'''

import random
import sys
import argparse
import logging
import os
import time

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, hamming_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup)

from utils import data_helper as helper
from utils.data_loader import CodeDataset
from models.exception_network import ExceptionLSTMNet, ExceptionAttentionNet


logger = logging.getLogger(__name__)


def train(args, tokenizer, encoder):

    train_dataset = CodeDataset(args.X_train, args.y_train, args.code_length, tokenizer,)
    valid_dataset = CodeDataset(args.X_val, args.y_val, args.code_length, tokenizer)
    test_dataset = CodeDataset(args.X_test, args.y_test, args.code_length, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, shuffle=True, drop_last=True)
    
    # model = ExceptionNet(n_classes=args.y_train.shape[1], encoder=encoder, embedding_size=args.embedding_size,lstm_hidden_size=256, fc_hidden_size=256, drop_out=0.5)
    model = ExceptionAttentionNet(n_classes=args.y_train.shape[1], encoder=encoder, embedding_size=args.embedding_size, 
                        heads=args.heads, num_layers=args.num_layers, forward_expansion=args.forward_expansion, drop_out=0.5)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,num_training_steps=len(train_loader)*args.num_train_epochs)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        print("Let's use", args.n_gpu, "GPUs!")
        model = nn.DataParallel(model)
    model.to(args.device)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
    logger.info("  Total train batch size  = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", len(train_loader)*args.num_train_epochs)

    model.zero_grad()
    
    model.train()
    tr_num,tr_loss,args.best_auprc=0,0,0
    print_every = 100
    for idx in range(args.num_train_epochs): 
        for step,batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(args.device)
            attention_mask = batch['attention_mask'].to(args.device)
            labels = batch['labels'].to(args.device)
            
            output = model(input_ids, attention_mask)
            loss = criterion(output, labels)
            
            tr_loss += loss.item()
            tr_num+=1
            if (step+1) % print_every == 0:
                print("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss=0
                tr_num=0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()


        # Eval!
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Valid Batch size = %d", args.valid_batch_size)

        model.eval()
        valid_num,valid_loss = 0, 0
        true_onehot_labels, predicted_onehot_labels = [], []
        predicted_onehot_scores = []
        predicted_onehot_labels_tk = [[] for _ in range(args.top_num)]
        eval_pre_tk = [0.0 for _ in range(args.top_num)]

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                input_ids = batch['input_ids'].to(args.device)
                attention_mask = batch['attention_mask'].to(args.device)
                labels = batch['labels'].to(args.device)

                output = model(input_ids, attention_mask)
                loss = criterion(output, labels)

                valid_loss += loss.item() 
                valid_num += args.valid_batch_size

                # Prepare for calculating metrics
                for label in labels:
                    true_onehot_labels.append(label.tolist())
                for out in output:
                    predicted_onehot_scores.append(out.tolist())

                # Predict by threshold
                batch_predicted_onehot = helper.get_onehot_label_threshold(output=output.cpu().detach().numpy(), threshold=0.5)
                for onehot_labels in batch_predicted_onehot:
                    predicted_onehot_labels.append(onehot_labels)

                # Predict by topK
                for num in range(args.top_num):
                    batch_predicted_onehot_labels_tk = \
                            helper.get_onehot_label_topk(scores=output.cpu().detach().numpy(), top_num=num + 1)
                    for onehot_labels in batch_predicted_onehot_labels_tk:
                        predicted_onehot_labels_tk[num].append(onehot_labels)

            print("epoch {} step {} val loss {}".format(idx,step+1,round(valid_loss/valid_num,5)))
                    
            # Calculate Precision & Recall & F1
            eval_pre_ts = precision_score(y_true=np.array(true_onehot_labels),
                                                y_pred=np.array(predicted_onehot_labels), average='micro')
            eval_rec_ts = recall_score(y_true=np.array(true_onehot_labels),
                                            y_pred=np.array(predicted_onehot_labels), average='micro')
            eval_F_ts = f1_score(y_true=np.array(true_onehot_labels),
                                        y_pred=np.array(predicted_onehot_labels), average='micro')
                    
            eval_hamming = hamming_loss(y_true=np.array(true_onehot_labels),y_pred=np.array(predicted_onehot_labels))

            # Calculate the average AUC
            eval_auc = roc_auc_score(y_true=np.array(true_onehot_labels),
                                            y_score=np.array(predicted_onehot_scores), average='micro')
            # Calculate the average PR
            eval_prc = average_precision_score(y_true=np.array(true_onehot_labels),
                                                    y_score=np.array(predicted_onehot_scores), average='micro')

            for num in range(args.top_num):
                eval_pre_tk[num] = precision_score(y_true=np.array(true_onehot_labels),
                                                       y_pred=np.array(predicted_onehot_labels_tk[num]), average='micro')      
            logger.info("All Validation set: AUC {0:g} | AUPRC {1:g}"
                                .format(eval_auc, eval_prc))
            logger.info("Predict by threshold: Precision {0:g}, Recall {1:g}, F {2:g}, Hamming Loss {3:g}"
                                .format(eval_pre_ts, eval_rec_ts, eval_F_ts, eval_hamming))

            for num in range(args.top_num):
                logger.info("Top{0}: Precision {1:g}"
                                .format(num + 1, eval_pre_tk[num]))

        model.train()
        
        #save best model
        logger.info("Saving Model with best parameters")
        if eval_prc > args.best_auprc:
            args.best_auprc = eval_prc
            
            logger.info("  "+"*"*20)
            logger.info("  Best AUPRC:%s",round(args.best_auprc,4))
            logger.info("  "+"*"*20)

            timestamp = str(int(time.time()))
            checkpoint_prefix = 'checkpoint-best-auprc'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            helper.save_checkpoint({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_auprc': args.best_auprc,
                },filename=os.path.join(output_dir, "epoch%d.%s.pth" % (idx, timestamp)))
        


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.deterministic = False

'''
python train.py --data_file multi_label_leaf_43.csv --output_dir saved_models/model --num_train_epochs 20 --train_batch_size 100 --valid_batch_size 100 --top_num 1
'''
def main():
    # multi_label_leaf_43

    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_file", default=None, type=str, required=True,
                        help="The input training data file (a json file).")
    parser.add_argument("--output_dir", default=os.path.abspath(os.curdir), type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    
    parser.add_argument("--model_name_or_path", default="microsoft/codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=512, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    
    # parser.add_argument("--do_train", action='store_true',
    #                     help="Whether to run training.")
    # parser.add_argument("--do_eval", action='store_true',
    #                     help="Whether to run eval on the dev set.")
    # parser.add_argument("--do_test", action='store_true',
    #                     help="Whether to run eval on the test set.")  
    

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--valid_batch_size", default=4, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=2, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--heads", default=4, type=int,
                        help="Number of head.")
    parser.add_argument("--num_layers", default=2, type=int,
                        help="Number of layer.")
    parser.add_argument("--forward_expansion", default=4, type=int,
                        help="Forward expansion.")

    parser.add_argument('--top_num', type=int, default=1,
                        help="Recommend k")                        
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    # Print arguments
    args = parser.parse_args()

    # Set log
    # logging.basicConfig(
    #     filename='./saved_models/logs/catching_exception-torch.log',
    #     datefmt='%m/%d/%Y %H:%M:%S',
    #     level=logging.INFO,
    #     filemode='w',
    #     format='%(name)s - %(levelname)s - %(message)s')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s",device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # Build pretrained encoder
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = RobertaTokenizer.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    encoder = RobertaModel.from_pretrained(args.model_name_or_path)
    encoder.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    X, y = helper.load_data(os.path.abspath(os.curdir)+os.sep+'dataset'+os.sep+args.data_file)

    train_size = 0.8
    test_size = 0.5
    val_size = 1 - test_size

    # Perform Multi-label stratified sampling
    X_train, X_, y_train, y_ = helper.iterative_train_test_split(X, y, split_size=train_size)
    X_val, X_test, y_val, y_test = helper.iterative_train_test_split(X_, y_, split_size=val_size)
    args.X_train, args.y_train = X_train, y_train
    args.X_val, args.y_val = X_val, y_val
    args.X_test, args.y_test = X_test, y_test

    args.embedding_size = config.hidden_size

    train(args, tokenizer, encoder)


if __name__ == '__main__':
    main()

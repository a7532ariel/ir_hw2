import torch
import pandas as pd
import numpy as np
import logging
import os 
import argparse

import wandb
with open("API_KEY", "r") as f:
    KEY = f.read()
    wandb.login(key=KEY)

loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')

from torch import nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from model import BPRModel
from utils import bpr_loss, calc_map

np.random.seed(17)
torch.manual_seed(17)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument("--lr", 
	type=float, 
	default=0.0001, 
	help="learning rate")
parser.add_argument("--lamda", 
	type=float, 
	default=0, 
	help="model regularization rate")
parser.add_argument("--dropout", 
	type=float, 
	default=0.3, 
	help="model regularization rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=64, 
	help="batch size for training")
parser.add_argument("--epoch", 
	type=int,
	default=60,  
	help="training epoches")
parser.add_argument("--top_k", 
	type=int, 
	default=50, 
	help="compute metrics@top_k")
parser.add_argument("--factor", 
	type=int,
	default=128, 
	help="predictive factors numbers in the model")
parser.add_argument("--do_train", 
	default=False, 
	help="predictive factors numbers in the model")
parser.add_argument("--do_test", 
	default=False, 
	help="predictive factors numbers in the model")
parser.add_argument("--model", 
	default='', 
	help="predictive factors numbers in the model")
parser.add_argument("--save", 
	default='result.csv', 
	help="predictive factors numbers in the model")

parser.add_argument("--save_dir", 
	default='.', 
	help="predictive factors numbers in the model")

args = parser.parse_args()
wandb.init(project="irhw2")
wandb.config.update(args)

batch_size = args.batch_size
epoch = args.epoch
lr = args.lr
weight_decay = args.lamda
factor = args.factor

def bpr_convert_data_to_positive_feature(data, ratio=0.11):
    train_feature = []
    train_user_positive = []
    valid_user_positive = []
    
    
    for i, positive_set in enumerate(data):
        valid_positives = []
        train_positives = []
        
        x = np.arange(positive_set.shape[0])
        np.random.shuffle(x)
        train_positive_set = positive_set[x[int(positive_set.shape[0]*ratio):]]
        valid_positive_set = positive_set[x[:int(positive_set.shape[0]*ratio)]]
        for j in train_positive_set:
            train_feature.append({
                'user': i,
                'positive': j,
            })
            train_positives.append(j)
        for j in valid_positive_set:
            valid_positives.append(j)
            
        train_user_positive.append(train_positives)
        valid_user_positive.append(valid_positives)
    return train_feature, valid_user_positive, train_user_positive


def negative_sampling(users, negative_matrix, pos=1):
    negatives = []
    for user in users:
        negative_items = negative_matrix[user]
        negatives.append(np.random.choice(negative_items, 1, replace=False)[0])
    return torch.tensor(negatives, dtype=torch.long).to(device)

if __name__ == "__main__":
    train = pd.read_csv('train.csv')
    users = train['UserId'].values
    num_users = len(users)

    positive_items = train['ItemId'].values
    max_item = 0
    for i, row in enumerate(positive_items):
        row_int = [int(x) for x in row.split(' ')]
        positive_items[i] = np.array(row_int)
        max_item = max(max_item, np.max(positive_items[i]))
    num_items = max_item+1

    negative_matrix = []
    for i, row in enumerate(positive_items):
        negtive_items = np.delete(np.arange(num_items), row)
        negative_matrix.append(negtive_items)
    
    users_tensor = torch.tensor(users, dtype=torch.long).to(device)
    items_tensor = torch.tensor(np.arange(num_items), dtype=torch.long).to(device)
        
    if args.do_train: 
        train_feature, valid_feature, train_user_positive = bpr_convert_data_to_positive_feature(positive_items)

        train_users = torch.tensor([f['user'] for f in train_feature], dtype=torch.long)
        train_positives = torch.tensor([f['positive'] for f in train_feature], dtype=torch.long)

        train_dataset = TensorDataset(train_users, train_positives)
        train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size=batch_size)

        model = BPRModel(num_users, num_items, n_factors=factor, dropout_p=args.dropout)
        model.to(device)
        
        wandb.watch(model)
        
        optimizer = torch.optim.SGD(model.parameters(),lr=lr)
        model.zero_grad()
        model.train()
        global_step = 0
        max_map = 0
        for ep in range(epoch):

            tr_loss = 0
            tr_step = 0
            model.train()
            for step, batch in enumerate(train_dataloader):
                model.zero_grad()
                tr_step += 1
                global_step += 1
                batch = tuple(t.to(device) for t in batch)
                users, positives = batch
                negatives = negative_sampling(users, negative_matrix)
                pred_positive, pred_negative = model(users, positives, negatives)
                loss = bpr_loss(pred_positive, pred_negative)
                tr_loss += loss.item()

                loss.backward()
                optimizer.step()

                if step%1000 == 0:
                    logging.info(f'Training loss: {tr_loss/tr_step/batch_size},  global step: {global_step}')
#                     wandb.log({"Training loss": tr_loss/tr_step/batch_size}, step=step)
            

            # valid 
            model.eval()
            train_avg_map, valid_avg_map = 0, 0
            dot = model.predict(users_tensor, items_tensor)

            for user in users_tensor:
                train_candidate_items = np.array(train_user_positive[user] + negative_matrix[user].tolist())
                train_scores = dot[user][train_candidate_items].cpu().detach().numpy()
                train_topk_indices = train_scores.argsort()[-args.top_k:][::-1]
                train_map_score = calc_map(train_candidate_items[train_topk_indices], train_user_positive[user])
                train_avg_map += train_map_score
                
                valid_candidate_items = np.array(valid_feature[user] + negative_matrix[user].tolist())
                valid_scores = dot[user][valid_candidate_items].cpu().detach().numpy()
                valid_topk_indices = valid_scores.argsort()[-args.top_k:][::-1]
                valid_map_score = calc_map(valid_candidate_items[valid_topk_indices], valid_feature[user])
                valid_avg_map += valid_map_score

            train_avg_map /= num_users
            valid_avg_map /= num_users    
            logging.info(f'EPOCH: {ep}, Train MAP: {train_avg_map}, Valid MAP: {valid_avg_map}')
            wandb.log({ 'epoch': ep, "Train MAP": train_avg_map, "Valid MAP": valid_avg_map, 'Training loss': tr_loss/tr_step/batch_size })

            if valid_avg_map > max_map:
                max_map = valid_avg_map
                torch.save(model, f'{args.save_dir}/ckpt_{valid_avg_map}.model')
                logging.info(f'saving model with valid map {valid_avg_map}')
                
    if args.do_test:
        if not args.do_train:
            model = torch.load(args.model)
        model.eval()
        
        dot = model.predict(users_tensor, items_tensor)
        result = []
        for user in users_tensor:
            candidate_items = negative_matrix[user]
            scores = dot[user][candidate_items].cpu().detach().numpy()
            topk_indices = scores.argsort()[-args.top_k:][::-1]
            
            topk = candidate_items[topk_indices].tolist()
            topk = [str(x) for x in topk]
            
            result.append(' '.join(topk))
        
        
        df = pd.DataFrame({'UserId':users,'ItemId':result})
        df.to_csv(args.save, index=False, sep=',')
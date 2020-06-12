import torch
from torch import nn

class BaseModel(nn.Module):
    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=32,
                 dropout_p=0.3):
        super(BaseModel, self).__init__()
        
        self.num_users = n_users
        self.num_items = n_items
        
        self.user_biases = nn.Embedding(n_users, 1)
        self.item_biases = nn.Embedding(n_items, 1)
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        
        self.dropout = nn.Dropout(dropout_p)
        
    
    def forward(self, users, items):
        user_embed = self.user_embeddings(users) # 1xf
        item_embed = self.item_embeddings(items) # 1xf
        
        dot = (self.dropout(user_embed) * self.dropout(item_embed)).sum(dim=-1, keepdim=True)
        
#         pred = dot + self.user_biases(users) + self.item_biases(items)

        return dot


class BPRModel(nn.Module):
    def __init__(self,
                 n_users,
                 n_items,
                 n_factors=32,
                 dropout_p=0.3):
        super(BPRModel, self).__init__()
        self.num_users = n_users
        self.num_items = n_items
#         self.model = basemodel(n_users, n_items, n_factors=n_factors, dropout_p=dropout_p)
        
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        
#         nn.init.normal_(self.user_embeddings.weight, std=0.01)
#         nn.init.normal_(self.item_embeddings.weight, std=0.01)
        
        self.dropout = nn.Dropout(dropout_p)
        
    
    def forward(self, users, positives, negatives):
        user_embed = self.user_embeddings(users) 
        positive_embed = self.item_embeddings(positives)
        negative_embed = self.item_embeddings(negatives)
        
        pred_positive = (user_embed * positive_embed).sum(dim=-1)
        pred_negative = (user_embed * negative_embed).sum(dim=-1)
        
        return pred_positive, pred_negative
    
    def predict(self, user, items):
        user_embed = self.user_embeddings(user)
        
        product = torch.mm(user_embed, self.item_embeddings(items).transpose(0, 1))
        
        return product
        



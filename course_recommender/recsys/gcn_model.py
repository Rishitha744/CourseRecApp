import math
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# ---------------------------
# Dataset & preprocessing
# ---------------------------
class TriplesDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return self.triples[idx]

def load_triples_from_csv(path, user_col, item_col, rating_col, sep=","):
    df = pd.read_csv(path, sep=sep)
    df[user_col] = df[user_col].astype(str)
    df[item_col] = df[item_col].astype(str)
    df[rating_col] = df[rating_col].astype(float).round().clip(1,5).astype(int)

    user2idx = {u:i for i,u in enumerate(df[user_col].unique())}
    item2idx = {v:i for i,v in enumerate(df[item_col].unique())}

    triples = [(user2idx[row[user_col]], item2idx[row[item_col]], int(row[rating_col])) 
               for _,row in df.iterrows()]

    triples, user2idx, item2idx = remap_contiguous(triples)
    R = int(df[rating_col].nunique())
    return triples, user2idx, item2idx, R

def remap_contiguous(triples):
    users = sorted({t[0] for t in triples})
    items = sorted({t[1] for t in triples})
    usermap = {old:i for i,old in enumerate(users)}
    itemmap = {old:j for j,old in enumerate(items)}
    new_triples = [(usermap[u], itemmap[v], r) for (u,v,r) in triples]
    return new_triples, usermap, itemmap

def build_edge_lists(triples, num_users, num_items, R):
    lists_u = [[] for _ in range(R)]
    lists_v = [[] for _ in range(R)]
    deg_user = np.zeros(num_users)
    deg_item = np.zeros(num_items)

    for (u,v,r) in triples:
        idx = r - 1
        lists_u[idx].append(u)
        lists_v[idx].append(v)
        deg_user[u] += 1
        deg_item[v] += 1

    edges = []
    for r in range(R):
        edges.append((
            torch.tensor(lists_u[r], dtype=torch.long),
            torch.tensor(lists_v[r], dtype=torch.long)
        ))

    return edges, torch.tensor(deg_user, dtype=torch.float32), torch.tensor(deg_item, dtype=torch.float32)

# ---------------------------
# Model
# ---------------------------
class GCMCEncoder(nn.Module):
    def __init__(self, num_users, num_items, in_dim, hid_dim, out_dim, R, 
                 node_dropout=0.0, hidden_dropout=0.5):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.R = R
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim

        self.user_input = nn.Embedding(num_users, in_dim)
        self.item_input = nn.Embedding(num_items, in_dim)
        nn.init.xavier_uniform_(self.user_input.weight)
        nn.init.xavier_uniform_(self.item_input.weight)

        self.W_item2user = nn.ParameterList([nn.Parameter(torch.empty(in_dim, hid_dim)) for _ in range(R)])
        self.W_user2item = nn.ParameterList([nn.Parameter(torch.empty(in_dim, hid_dim)) for _ in range(R)])

        for p in self.W_item2user:
            nn.init.xavier_uniform_(p)
        for p in self.W_user2item:
            nn.init.xavier_uniform_(p)

        self.W_out_user = nn.Linear(hid_dim, out_dim)
        self.W_out_item = nn.Linear(hid_dim, out_dim)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(hidden_dropout)

    def forward(self, edges_by_rating, deg_user, deg_item, mask_node_dropout=None):
        device = next(self.parameters()).device
        X_u = self.user_input.weight
        X_v = self.item_input.weight

        m_u = torch.zeros((self.num_users, self.hid_dim), device=device)
        m_v = torch.zeros((self.num_items, self.hid_dim), device=device)

        eps = 1e-10
        deg_u = torch.sqrt(torch.clamp(deg_user, min=eps))
        deg_v = torch.sqrt(torch.clamp(deg_item, min=eps))

        for r in range(self.R):
            u_idx, v_idx = edges_by_rating[r]
            if u_idx.numel() == 0:
                continue

            u_idx = u_idx.to(device)
            v_idx = v_idx.to(device)

            norm = 1.0 / (deg_u[u_idx] * deg_v[v_idx])

            msgs_to_u = (X_v[v_idx] @ self.W_item2user[r]) * norm.unsqueeze(1)
            m_u.index_add_(0, u_idx, msgs_to_u)

            msgs_to_v = (X_u[u_idx] @ self.W_user2item[r]) * norm.unsqueeze(1)
            m_v.index_add_(0, v_idx, msgs_to_v)

        h_u = self.dropout(self.act(self.W_out_user(m_u)))
        h_v = self.dropout(self.act(self.W_out_item(m_v)))

        return h_u, h_v

class BilinearDecoder(nn.Module):
    def __init__(self, emb_dim, R, nb=4):
        super().__init__()
        self.R = R
        self.nb = nb
        self.Ps = nn.Parameter(torch.empty(nb, emb_dim, emb_dim))
        nn.init.xavier_uniform_(self.Ps)
        self.a = nn.Parameter(torch.empty(R, nb))
        nn.init.xavier_uniform_(self.a)

    def forward(self, U, V, users_idx, items_idx):
        hu = U[users_idx]
        hv = V[items_idx]
        tmp = [(hu @ self.Ps[s] * hv).sum(dim=1) for s in range(self.nb)]
        tmp = torch.stack(tmp, dim=0)
        logits = (self.a @ tmp).t()
        return logits

class GCMC(nn.Module):
    def __init__(self, num_users, num_items, in_dim, hid_dim, emb_dim, R, nbasis=4,
                 node_dropout=0.2, hidden_dropout=0.5):
        super().__init__()
        self.encoder = GCMCEncoder(num_users, num_items, in_dim, hid_dim, emb_dim, R,
                                   node_dropout=node_dropout, hidden_dropout=hidden_dropout)
        self.decoder = BilinearDecoder(emb_dim, R, nb=nbasis)

    def forward(self, edges_by_rating, deg_user, deg_item, users_idx_batch, items_idx_batch):
        U, V = self.encoder(edges_by_rating, deg_user, deg_item)
        logits = self.decoder(U, V, users_idx_batch, items_idx_batch)
        return logits, U, V

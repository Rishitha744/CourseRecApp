import math
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import model + utilities from recsys app
from recsys.gcmc_model import (
    TriplesDataset,
    load_triples_from_csv,
    build_edge_lists,
    GCMC
)

# ---------------------------
# Safe batch helper
# ---------------------------
def safe_batch(batch, num_users, num_items, R):
    users = torch.tensor([b[0] for b in batch], dtype=torch.long)
    items = torch.tensor([b[1] for b in batch], dtype=torch.long)
    labels = torch.tensor([int(b[2]) for b in batch], dtype=torch.long)

    users = torch.clamp(users, 0, num_users - 1)
    items = torch.clamp(items, 0, num_items - 1)
    labels = torch.clamp(labels, 1, R)
    return users, items, labels

# ---------------------------
# Train one epoch
# ---------------------------
def train_epoch(model, optimizer, dataset_edges, edges_by_rating, deg_user, deg_item,
                batch_size=1024, device='cpu'):

    model.train()
    loader = DataLoader(dataset_edges, batch_size=batch_size, shuffle=True)
    ce = nn.CrossEntropyLoss()

    losses = []

    for batch in loader:
        batch = list(batch)
        users, items, labels = safe_batch(
            batch, model.encoder.num_users, model.encoder.num_items, model.decoder.R
        )
        users = users.to(device)
        items = items.to(device)
        labels_ce = (labels - 1).to(device)

        optimizer.zero_grad()

        logits, _, _ = model(
            edges_by_rating,
            deg_user.to(device),
            deg_item.to(device),
            users_idx_batch=users,
            items_idx_batch=items
        )

        loss = ce(logits, labels_ce)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return float(np.mean(losses))

# ---------------------------
# Evaluation (RMSE)
# ---------------------------
def evaluate(model, dataset_edges, edges_by_rating, deg_user, deg_item,
             batch_size=2048, device='cpu'):

    model.eval()
    loader = DataLoader(dataset_edges, batch_size=batch_size, shuffle=False)

    total = 0
    sum_sq_err = 0.0
    rating_values = torch.arange(1, model.decoder.R + 1, dtype=torch.float32, device=device)

    with torch.no_grad():
        for batch in loader:
            batch = list(batch)
            users, items, labels = safe_batch(
                batch, model.encoder.num_users, model.encoder.num_items, model.decoder.R
            )
            users = users.to(device)
            items = items.to(device)
            labels = labels.to(device).float()

            logits, _, _ = model(
                edges_by_rating,
                deg_user.to(device),
                deg_item.to(device),
                users_idx_batch=users,
                items_idx_batch=items
            )

            probs = F.softmax(logits, dim=1)
            expected = torch.sum(probs * rating_values.unsqueeze(0), dim=1)

            sum_sq_err += torch.sum((expected - labels)**2).item()
            total += labels.size(0)

    return math.sqrt(sum_sq_err / total)

# ---------------------------
# Main training logic
# ---------------------------
def main(args):

    print("Loading CSV dataset...")
    triples_idx, user2idx, item2idx, R = load_triples_from_csv(
        args.data, args.user_col, args.item_col, args.rating_col, sep=args.sep
    )

    num_users = len(user2idx)
    num_items = len(item2idx)

    random.shuffle(triples_idx)
    n = len(triples_idx)
    ntrain = int(n * 0.8)
    nval = int(n * 0.1)

    train_triples = triples_idx[:ntrain]
    val_triples   = triples_idx[ntrain:ntrain+nval]
    test_triples  = triples_idx[ntrain+nval:]

    print("Building graph...")
    edges_by_rating, deg_user, deg_item = build_edge_lists(
        train_triples, num_users, num_items, R
    )

    train_dataset = TriplesDataset(train_triples)
    val_dataset = TriplesDataset(val_triples)
    test_dataset = TriplesDataset(test_triples)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    print("Initializing GCMC model...")
    model = GCMC(
        num_users=num_users,
        num_items=num_items,
        in_dim=args.in_dim,
        hid_dim=args.hid_dim,
        emb_dim=args.emb_dim,
        R=R,
        nbasis=args.nbasis,
        node_dropout=args.node_dropout,
        hidden_dropout=args.hidden_dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_test = float("inf")

    print("\n======== TRAINING STARTED ========\n")

    for epoch in range(1, args.epochs + 1):

        train_loss = train_epoch(
            model, optimizer, train_dataset,
            edges_by_rating, deg_user, deg_item,
            batch_size=args.batch_size, device=device
        )

        val_rmse = evaluate(
            model, val_dataset, edges_by_rating, deg_user, deg_item,
            batch_size=args.batch_size, device=device
        )

        test_rmse = evaluate(
            model, test_dataset, edges_by_rating, deg_user, deg_item,
            batch_size=args.batch_size, device=device
        )

        print(f"Epoch {epoch:03d}  Train={train_loss:.4f}  Val={val_rmse:.4f}  Test={test_rmse:.4f}")

        # Save checkpoint after first epoch
        if epoch == 1:
            torch.save(model.state_dict(), "model_epoch1.pt")
            print("✔ Saved model weights: model_epoch1.pt")

            torch.save({
                "model_state": model.state_dict(),
                "user2idx": user2idx,
                "item2idx": item2idx,
                "R": R,
                "edges_by_rating": edges_by_rating,
                "deg_user": deg_user,
                "deg_item": deg_item
            }, "gcmc_full_epoch1.pt")
            print("✔ Saved full checkpoint: gcmc_full_epoch1.pt")

        if val_rmse < best_val:
            best_val = val_rmse
            best_test = test_rmse

    print(f"\nBest Val RMSE = {best_val:.4f} | Best Test RMSE = {best_test:.4f}\n")

# ---------------------------
# Args class
# ---------------------------
class Args:
    data = "course_ratings_dataset.csv"
    sep = ","
    user_col = "userid"
    item_col = "course_name"
    rating_col = "rating"

    in_dim = 64
    hid_dim = 64
    emb_dim = 32
    nbasis = 4
    lr = 0.01
    weight_decay = 1e-4
    epochs = 5
    batch_size = 2048
    node_dropout = 0.1
    hidden_dropout = 0.3
    cpu = True

# ---------------------------
# Run script
# ---------------------------
if __name__ == "__main__":
    args = Args()
    main(args)

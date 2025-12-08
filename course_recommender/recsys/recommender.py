import os
import torch
from django.conf import settings
from recsys.models import Course
from .gcmc_model import GCMC
from recsys.models import Rating, Course
from django.db.models import Q

# Load checkpoint
MODEL_PATH = os.path.join(settings.GCMC_MODEL_PATH, "gcmc_full_epoch1.pt")
checkpoint = torch.load(MODEL_PATH, map_location="cpu")

model_state = checkpoint["model_state"]
user2idx = checkpoint["user2idx"]
item2idx = checkpoint["item2idx"]
R = checkpoint["R"]

num_users = len(user2idx)
num_items = len(item2idx)

# Build model
model = GCMC(
    num_users=num_users,
    num_items=num_items,
    in_dim=64,
    hid_dim=64,
    emb_dim=32,
    R=R,
    nbasis=4,
    node_dropout=0.0,
    hidden_dropout=0.0
)

model.load_state_dict(model_state)
model.eval()

# Reverse lookup
idx2item = {v: k for k, v in item2idx.items()}

from django.db.models import Q, Case, When, IntegerField, F

def recommend_cold_start(interests, top_k=20):
    if not interests:
        return Course.objects.none()

    # split and clean
    keywords = [w.strip().lower() for w in interests.split(",") if w.strip()]

    # Build OR match query
    query = Q()
    for word in keywords:
        query |= Q(description__icontains=word) | Q(skills__icontains=word)

    qs = Course.objects.filter(query).distinct()

    # Add scoring — safe alias generation
    for word in keywords:
        safe_alias = "match_" + word.replace(" ", "_").replace("-", "_").replace("+", "")
        
        qs = qs.annotate(
            **{
                safe_alias: Case(
                    When(description__icontains=word, then=1),
                    When(skills__icontains=word, then=1),
                    default=0,
                    output_field=IntegerField(),
                )
            }
        )

    # collect all safe alias fields
    match_fields = [
        "match_" + word.replace(" ", "_").replace("-", "_").replace("+", "")
        for word in keywords
    ]

    # sum up relevance score
    qs = qs.annotate(
        relevance=sum(F(field) for field in match_fields)
    )

    # order by relevance
    qs = qs.order_by("-relevance")[:top_k]

    return qs

from django.db.models import Avg

def recommend_similarity(user, top_k=20):
    # Get this user's ratings
    user_ratings = Rating.objects.filter(user=user)

    # No ratings → return none (dashboard will handle cold start)
    if not user_ratings.exists():
        return Course.objects.none()

    # Find similar users who rated the same courses
    similar_users = Rating.objects.filter(
        course__in=user_ratings.values("course")
    ).exclude(user=user)

    # Aggregate rating score for courses liked by similar users
    course_scores = (
        similar_users.values("course__id", "course__name")
        .annotate(avg_rating=Avg("rating"))
        .order_by("-avg_rating")
    )

    # Extract top course IDs
    course_ids = [c["course__id"] for c in course_scores[:top_k]]

    # Return Course objects
    return Course.objects.filter(id__in=course_ids)


# Recommend courses
def recommend_courses(user_idx, top_k=20):
    if user_idx is None:
        return []

    with torch.no_grad():
        num_items = model.encoder.num_items

        # Create batch of all items
        item_batch = torch.arange(num_items, dtype=torch.long)

        # Repeat the same user index for all items
        user_batch = torch.full((num_items,), user_idx, dtype=torch.long)

        # ------- FORWARD PASS (correct signature) -------
        logits, U, V = model(
            checkpoint["edges_by_rating"],
            checkpoint["deg_user"],
            checkpoint["deg_item"],
            users_idx_batch=user_batch,
            items_idx_batch=item_batch
        )

        # ------- EXPECTED RATING COMPUTATION -------
        R = model.decoder.R  # e.g., 5
        rating_values = torch.arange(1, R + 1).float()  # [1,2,3,4,5]

        probs = torch.softmax(logits, dim=1)  # (num_items, R)
        expected_ratings = torch.sum(
            probs * rating_values.unsqueeze(0),
            dim=1
        )  # (num_items,)

        # ------- TOP-K -------
        top_items = torch.topk(expected_ratings, top_k).indices.tolist()

        # Fetch from DB while preserving ranking
        idx_order = {idx: pos for pos, idx in enumerate(top_items)}
        courses = list(Course.objects.filter(item_idx__in=top_items))
        courses.sort(key=lambda c: idx_order[c.item_idx])

        return courses

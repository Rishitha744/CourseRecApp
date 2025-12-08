from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm
from .forms import SignUpForm, OnboardingForm, SearchForm
from .models import Course, UserProfile
from .recommender import recommend_courses, recommend_similarity, recommend_cold_start
from .models import UserProfile
from django.contrib.auth.models import User
from .recommender import user2idx
from recsys.models import Rating, Course
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import OnboardingForm


def home_view(request):
    return render(request, "recsys/home.html")

def signup_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()

            username = user.username
            user_idx = user2idx.get(username, None)

            UserProfile.objects.create(
                user=user,
                user_idx=user_idx
            )

            login(request, user)
            return redirect("recsys:onboarding")
    else:
        form = UserCreationForm()

    return render(request, "recsys/signup.html", {"form": form})

@login_required
def onboarding_view(request):
    if request.method == "POST":
        form = OnboardingForm(request.POST)
        if form.is_valid():
            raw_text = form.cleaned_data["interests"].lower().strip()

            # Split interests by commas
            phrases = [p.strip() for p in raw_text.split(",") if p.strip()]

            # Remove duplicates (preserve order)
            unique_phrases = list(dict.fromkeys(phrases))

            # Save as comma-separated string
            interests = ",".join(unique_phrases)

            profile, _ = UserProfile.objects.get_or_create(user=request.user)
            profile.interests = interests
            profile.save()

            messages.success(request, "Your interests have been saved!")

            return redirect("recsys:dashboard")
    else:
        form = OnboardingForm()

    return render(request, "recsys/onboarding.html", {"form": form})



@login_required(login_url='/login/')
def dashboard_view(request):
    user = request.user
    profile, _ = UserProfile.objects.get_or_create(user=user)

    rating_count = Rating.objects.filter(user=user).count()

    print(f"User '{user.username}' has {rating_count} ratings.")
    print(f"User index: {profile.user_idx}")

    # 🌟 CASE 1 — New user (no model index)
    if profile.user_idx is None:

        # (A) No ratings yet → cold start only
        if rating_count == 0:
            recommended = recommend_cold_start(profile.interests, top_k=20)
            mode = "interests_based"

        # (B) 1–2 ratings → hybrid (similarity + cold start)
        elif rating_count <= 2:
            sim_recs = list(recommend_similarity(user, top_k=10))
            cold_recs = list(recommend_cold_start(profile.interests, top_k=10))

            # merge lists but avoid duplicates
            combined = sim_recs + [c for c in cold_recs if c not in sim_recs]

            recommended = combined[:20]
            mode = "hybrid"

        # (C) 3+ ratings → similarity only
        else:
            recommended = recommend_similarity(user, top_k=20)
            mode = "similarity_based"

        return render(request, "recsys/dashboard.html", {
            "recommended": recommended,
            "mode": mode
        })

    # 🌟 CASE 2 — Existing user → model-based
    recommended = recommend_courses(profile.user_idx, top_k=20)
    return render(request, "recsys/dashboard.html", {
        "recommended": recommended,
        "mode": "model_based"
    })


@login_required
def rate_course(request, course_id, rating_value):
    course = Course.objects.get(id=course_id)
    Rating.objects.update_or_create(
        user=request.user,
        course=course,
        defaults={"rating": rating_value}
    )
    return redirect("recsys:search")



@login_required
def search_view(request):
    results = []
    searched = False

    # POST → bulk rating submission
    if request.method == "POST":
        for key, value in request.POST.items():
            if key.startswith("rating_") and value.isdigit():
                course_id = key.split("_")[1]
                Rating.objects.update_or_create(
                    user=request.user,
                    course_id=course_id,
                    defaults={"rating": int(value)}
                )
        return redirect("recsys:dashboard")

    # GET → search
    if request.method == "GET" and "q" in request.GET:
        q = request.GET.get("q", "").strip()
        if q:
            searched = True
            form = SearchForm(initial={"q": q})

            # unique course names
            names = (
                Course.objects.filter(name__icontains=q)
                .order_by("name")
                .values("name")
                .distinct()[:10]
            )

            # one real object per unique name
            results = [
                Course.objects.filter(name=n["name"]).first()
                for n in names
            ]
        else:
            form = SearchForm()
    else:
        form = SearchForm()

    return render(request, "recsys/search.html", {
        "form": form,
        "results": results,
        "searched": searched
    })

def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect("recsys:dashboard")
    else:
        form = AuthenticationForm()

    return render(request, "recsys/login.html", {"form": form})

from django.contrib.auth import logout

def logout_view(request):
    logout(request)
    return redirect("recsys:home")


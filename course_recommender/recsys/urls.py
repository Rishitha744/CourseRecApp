from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

app_name = "recsys"

urlpatterns = [
    path("", views.home_view, name="home"),
    path("signup/", views.signup_view, name="signup"),
    path("login/", auth_views.LoginView.as_view(
        template_name="recsys/login.html"
    ), name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("onboarding/", views.onboarding_view, name="onboarding"),
    path("search/", views.search_view, name="search"),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("rate/<int:course_id>/<int:rating_value>/", views.rate_course, name="rate_course"),
]

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)
    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

class OnboardingForm(forms.Form):
    interests = forms.CharField(
        label="Enter your interests (comma separated):",
        widget=forms.TextInput(attrs={
            "placeholder": "e.g. python, machine learning, data science"
        }),
        required=False
    )

class SearchForm(forms.Form):
    q = forms.CharField(label="Search courses", max_length=200)

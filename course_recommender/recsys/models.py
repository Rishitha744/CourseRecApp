# Create your models here.
from django.db import models
from django.contrib.auth.models import User

class Course(models.Model):
    item_idx = models.IntegerField(unique=True)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    skills = models.TextField(blank=True)

    def __str__(self):
        return self.name

class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    rating = models.IntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "course")

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    user_idx = models.IntegerField(null=True, blank=True)
    interests = models.TextField(blank=True)

    def __str__(self):
        return self.user.username

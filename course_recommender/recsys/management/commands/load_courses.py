from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from recsys.models import UserProfile, Course, Rating
from django.conf import settings
import csv
from pathlib import Path

class Command(BaseCommand):
    help = "Load users, courses and ratings from CSV into Django database"

    def handle(self, *args, **kwargs):
        csv_path = Path(settings.BASE_DIR) / "course_ratings_dataset.csv"

        if not csv_path.exists():
            self.stderr.write(f"CSV file not found at: {csv_path}")
            return

        self.stdout.write("Loading users, courses, and ratings...\n")

        user_index_map = {}   # maps CSV userid string -> numeric index
        next_user_idx = 0

        course_index_map = {}  # maps course name -> item_idx
        next_course_idx = 0

        total_rows = 0

        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for row in reader:
                total_rows += 1

                csv_user_id = row["userid"]  # keep as string
                course_name = row["course_name"]
                course_desc = row.get("course_description", "")
                rating_value = int(row["rating"])

                # Assign numeric user_idx if first time seen
                if csv_user_id not in user_index_map:
                    user_index_map[csv_user_id] = next_user_idx
                    next_user_idx += 1

                user_idx = user_index_map[csv_user_id]

                # Create Django user
                django_username = f"user_{csv_user_id}"
                user, created_user = User.objects.get_or_create(
                    username=django_username,
                    defaults={"password": "password123"}
                )

                # Create or update profile with numeric user_idx
                UserProfile.objects.update_or_create(
                    user=user,
                    defaults={"user_idx": user_idx}
                )

                # Assign item_idx for course
                if course_name not in course_index_map:
                    course_index_map[course_name] = next_course_idx
                    next_course_idx += 1

                item_idx = course_index_map[course_name]

                # Create course if needed
                course, created_course = Course.objects.get_or_create(
                    item_idx=item_idx,
                    defaults={"name": course_name, "description": course_desc}
                )

                # Create rating
                Rating.objects.get_or_create(
                    user=user,
                    course=course,
                    defaults={"rating": rating_value}
                )

                # Progress output every 500 rows
                if total_rows % 10 == 0:
                    self.stdout.write(f"Processed {total_rows} rows...")

        self.stdout.write(self.style.SUCCESS(
            f"\nDONE! Loaded {total_rows} rows, {next_user_idx} users, {next_course_idx} courses."
        ))

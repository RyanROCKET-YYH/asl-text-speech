from django.db import models
from django.contrib.auth.models import User


# Create your models here.
class Video(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video_file = models.FileField(upload_to='videos/')    # file field to store the uploaded video
    transcipt = models.TextField(blank=True)    # text field to store the generated transcript
    created_at = models.DateTimeField(auto_now_add=True)    # date time field to store the date and time of video upload

class Translation(models.Model):
    TRANSLATION_DIRECTIONS = [
        ('AT', 'ASL-to-Text'),
        ('TA', 'Text-to-ASL'),
    ]
    
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video = models.ForeignKey(Video, on_delete=models.CASCADE, blank=True, null=True)
    text = models.TextField(blank=True)
    direction = models.CharField(max_length=2, choices=TRANSLATION_DIRECTIONS)
    created_at = models.DateTimeField(auto_now_add=True)
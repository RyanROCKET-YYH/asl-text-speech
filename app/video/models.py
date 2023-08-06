from django.db import models
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.db.models.signals import post_delete
from django.dispatch import receiver

def validate_file_extension(value):
    import os
    ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
    valid_extensions = ['.mp4']
    if not ext.lower() in valid_extensions:
        raise ValidationError('Unsupported file extension. Upload an mp4 file.')
    
def user_directory_path(instance, filename):
    # file will be uploaded to MEDIA_ROOT/videos/user_<id>/<filename>
    return 'videos/user_{0}/{1}'.format(instance.user.id, filename)

class Video(models.Model):
    STATUS_CHOICES = [
        ('UPLOADED', 'Uploaded'),
        ('PROCESSING', 'Processing'),
        ('COMPLETED', 'Completed'),
        ('FAILED', 'Failed'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    video_file = models.FileField(upload_to=user_directory_path, validators=[validate_file_extension])   # file field to store the uploaded video
    transcript = models.TextField(blank=True)    # text field to store the generated transcript
    created_at = models.DateTimeField(auto_now_add=True)    # date time field to store the date and time of video upload
    processed_video_file = models.FileField(upload_to=user_directory_path, blank=True, null=True)   # file field to store the processed video
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='UPLOADED')  # status field to store the current status of the video

@receiver(post_delete, sender=Video)
def submission_delete(sender, instance, **kwargs):
    instance.video_file.delete(False)  # Delete the video file

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
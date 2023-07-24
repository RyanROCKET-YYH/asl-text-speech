from django import forms
from .models import Video

class VideoForm(forms.Form):
    video = forms.FileField()



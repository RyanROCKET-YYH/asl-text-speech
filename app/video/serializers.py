from rest_framework import serializers
from .models import Video, Translation

class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = Video
        fields = ['id', 'user', 'video_file','transcipt', 'created_at']

class TranslationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Translation
        fields = ['id', 'user', 'video', 'direction', 'text', 'created_at']

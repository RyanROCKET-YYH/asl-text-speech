from django.shortcuts import render
from rest_framework import generics
from .models import Video, Translation
from .serializers import VideoSerializer, TranslationSerializer
# Create your views here.
class VideoListCreateView(generics.ListCreateAPIView):
    queryset = Video.objects.all()
    serializer_class = VideoSerializer

class TranslationListCreateView(generics.ListCreateAPIView):
    queryset = Translation.objects.all()
    serializer_class = TranslationSerializer



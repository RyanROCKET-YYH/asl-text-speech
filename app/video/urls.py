
from django.urls import path, re_path
from .views import *
from django.contrib import admin
urlpatterns = [
    path('videos/', VideoListCreateView.as_view()),
    path('translations/', TranslationListCreateView.as_view()),
]

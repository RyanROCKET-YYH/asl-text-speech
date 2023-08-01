
from django.urls import path, re_path
from .views import *
from django.contrib import admin
urlpatterns = [
    path('run_ml_model/', RunMLModelView.as_view(), name='run_ml_model'),
    path('videos/', VideoListCreateView.as_view(), name='video-list-create'),
    path('upload/', VideoUploadView.as_view(), name='upload_video'),
    path('translations/', TranslationListCreateView.as_view(), name='translation-list-create'),
    path('run-script/', run_script, name='run-script'),
    path('run-script2/', run_script2, name='run-script2'),
    path('list/', VideoListView.as_view(), name='list_videos'),
    path('process/<int:video_id>/', process_video, name='process_video'),
]

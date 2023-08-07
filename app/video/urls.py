
from django.urls import path, re_path
from .views import *
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('videos/', VideoListCreateView.as_view(), name='video-list-create'),
    path('upload/', VideoUploadView.as_view(), name='upload_video'),
    path('translations/', TranslationListCreateView.as_view(), name='translation-list-create'),
    path('run-script-live/', run_script_live, name='run-script-live'),
    path('run-script2-live/', run_script2, name='run-script2-live'),
    path('list/', VideoListView.as_view(), name='list_videos'),
    path('process/<int:video_id>/', process_video, name='process_video'),
    path('process2/<int:video_id>/', process_video_2, name='process_video_2'),
    path('videos/<int:video_id>/delete/', delete_video, name='delete_video'),
    path('video/<int:video_id>/transcript/', view_transcript, name='view_transcript'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
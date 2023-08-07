
from django.urls import path, re_path
from .views import *
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('videos/', VideoListCreateView.as_view(), name='video-list-create'),
    path('upload/', VideoUploadView.as_view(), name='upload_video'),
    path('translations/', TranslationListCreateView.as_view(), name='translation-list-create'),
    # words-live is for wei's live translation
    path('run-script-words-live/', run_script_words_live, name='run-script-words-live'),
    # alphabets-live is for leiran's live translation
    path('run-script-alphabets-live/', run_script_alphabets_live, name='run-script-alphabets-live'),
    path('list/', VideoListView.as_view(), name='list_videos'),
    # process_words is for wei's pre-recorded video translation
    path('process_words/<int:video_id>/', process_video_words, name='process_video_words'),
    # process_alphabets is for leiran's pre-recorded video translation
    path('process_alphabets/<int:video_id>/', process_video_alphabets, name='process_video_alphabets'),
    path('videos/<int:video_id>/delete/', delete_video, name='delete_video'),
    # get translated result for wei's part
    path('video/<int:video_id>/transcript_words/', view_transcript_words, name='view_transcript_words'),
    # get translated result for leiran's part
    path('video/<int:video_id>/transcript_alphabets/', view_transcript_alphabets, name='view_transcript_alphabets'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
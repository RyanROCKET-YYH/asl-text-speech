from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views
from django.contrib.auth.views import LogoutView


urlpatterns = [
    path('admin/', admin.site.urls),
    path('users/', include('user.urls')), 
    path('videos/', include('video.urls')),
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('contact/', views.contact, name='contact'),
    path('page-404/', views.error_404, name='error_404'),
    path('text2asl/', views.text2asl, name='text2asl'),
    path('transcript/', views.transcript, name='transcript'),
    path('liveAlpha/', views.liveAlpha, name='liveAlpha'),
    path('liveWords/', views.liveWords, name='liveWords'),
    path('live/', views.live, name='live'),
    path('camera/', views.camera, name='camera'),
    path('home/', views.home, name='home'),
    path('list/', views.list, name='list'),
    path('upload/', views.upload, name='upload'),
    path('transcript_alpha/', views.run_script_alphabets_live_, name='transcript_alpha'),
    path('transcript_words/', views.run_script_words_live_, name='transcript_words'),
    path('logout/', LogoutView.as_view(next_page='index'), name='logout'),
]

urlpatterns = urlpatterns + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

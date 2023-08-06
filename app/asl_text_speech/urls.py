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
    path('logout/', LogoutView.as_view(next_page='index'), name='logout'),
    # path('receive_frame/', include('receive_frame.urls'), name='receive_frame'),

]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

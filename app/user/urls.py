
from django.urls import path, re_path
from .views import *

urlpatterns = [
    path('users/', UserListCreateView.as_view()),
    path('profiles/', UserProfileListCreateView.as_view()),
]

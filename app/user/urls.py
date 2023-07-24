
from django.urls import path, re_path
from .views import *

urlpatterns = [
    path('users/', UserListCreateView.as_view()),
    path('profiles/', UserProfileListCreateView.as_view(), name= 'profile_view'),
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
]

from django.urls import path
from .views import receive_image

urlpatterns = [
    path('', receive_image),
]

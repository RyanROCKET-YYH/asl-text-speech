from django.shortcuts import render, HttpResponse
from rest_framework import generics
from .models import User, UserProfile
from .serializers import UserSerializer, UserProfileSerializer

# Create your views here.
def login(request):
    html = '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
    </head>
    <body>
        User Name:<input type="text" name="uname" id=""> <br/>
        Password:<input type="password" name="pwd" id=""> <br/>
        <input type="submit" value="Login">
    </body>
    </html>
    '''
    return HttpResponse(html)

class UserListCreateView(generics.ListCreateAPIView):
    queryset = User.objects.all()
    serializer_class = UserSerializer

class UserProfileListCreateView(generics.ListCreateAPIView):
    queryset = UserProfile.objects.all()
    serializer_class = UserProfileSerializer
from django.http import HttpResponse

from django.shortcuts import render

def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def error_404(request):
    return render(request, 'page-404.html')

def camera_view(request):
    return render(request, 'camera.html')

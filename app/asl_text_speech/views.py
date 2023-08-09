from django.http import HttpResponse
from video.views import run_script_words_live, run_script_alphabets_live


from django.shortcuts import render

def run_script_words_live_(request):
    output, error = run_script_words_live(request)
    return output, error

def run_script_alphabets_live_(request):
    output, error = run_script_alphabets_live(request)
    return output, error

def index(request):
    return render(request, 'index.html')

def about(request):
    return render(request, 'about.html')

def contact(request):
    return render(request, 'contact.html')

def error_404(request):
    return render(request, 'page-404.html')

def text2asl(request):
    return render(request, 'text2asl.html')

def transcript(request):
    return render(request, 'transcript.html')

def camera(request):
    return render(request, 'camera.html')

def home(request):
    return render(request, 'home.html')

def live(request):
    return render(request, 'live.html')

def liveAlpha(request):
    return render(request, 'liveAlpha.html')

def liveWords(request):
    return render(request, 'liveWords.html')

def list(request):
    return render(request, 'video/list.html')

def upload(request):
    return render(request, 'video/upload.html')

def transcript_alpha(request):
    return render(request, 'video/transcript_alphabets.html')

def transcript_words(request):
    return render(request, 'video/transcript_words.html')
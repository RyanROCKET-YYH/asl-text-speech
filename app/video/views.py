from rest_framework import generics
from .models import Video, Translation
from .serializers import VideoSerializer, TranslationSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from .WLASL_Inference.Inference import process_sequence, load_model
import pandas as pd
import torch
import subprocess
from django.http import JsonResponse, HttpResponseRedirect
import os
from django.shortcuts import render, redirect
from .forms import VideoForm
from django.views import View
from .forms import VideoForm
from django.urls import reverse
from django.contrib import messages


# Create your views here.
class VideoListCreateView(generics.ListCreateAPIView):
    queryset = Video.objects.all()
    serializer_class = VideoSerializer

    def get_queryset(self):
        # Return only the videos uploaded by the current user
        return Video.objects.filter(user=self.request.user)

class TranslationListCreateView(generics.ListCreateAPIView):
    queryset = Translation.objects.all()
    serializer_class = TranslationSerializer

num_classes = 100
file = pd.read_csv("video/WLASL_Inference/dataset/wlasl_class_list.txt", sep='\t', header=None)
all_words = file[1].tolist()
glosses = all_words[:num_classes]

weights = 'video/WLASL_Inference/weights/nslt_100.pt'
i3d = load_model(weights, num_classes)

class RunMLModelView(APIView):
    def post(self, request, format=None):
        video = request.FILES['video']
        # Process video with ML model
        result = process_sequence(video, i3d, glosses, num_classes)
        return Response({'result': result})

def run_script(request):
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    script_directory = os.path.join(current_directory, 'WLASL_Inference')
    script_path = os.path.join(script_directory, 'Inference.py')

    try:
        # Change the working directory to the location of the script
        os.chdir(script_directory)

        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        output = result.stdout
        error = result.stderr
        response = {
            'output': output,
            'error': error
        }
        return JsonResponse(response)
    except subprocess.CalledProcessError as e:
        response = {
            'error': str(e)
        }
        return JsonResponse(response, status=500)
    finally:
        # Change the working directory back to its original location
        os.chdir(current_directory)

def run_script2(request):
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    script_directory = os.path.join(current_directory, 'alphabets')
    script_path = os.path.join(script_directory, 'main.py')

    try:
        # Change the working directory to the location of the script
        os.chdir(script_directory)

        result = subprocess.run(['python', script_path], capture_output=True, text=True)
        output = result.stdout
        error = result.stderr
        response = {
            'output': output,
            'error': error
        }
        return JsonResponse(response)
    except subprocess.CalledProcessError as e:
        response = {
            'error': str(e)
        }
        return JsonResponse(response, status=500)
    finally:
        # Change the working directory back to its original location
        os.chdir(current_directory)

class VideoUploadView(View):
    def get(self, request):
        form = VideoForm()
        return render(request, 'video/upload.html', {'form': form})

    def post(self, request, *args, **kwargs):
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            new_video = Video(video_file=request.FILES['video'], user=request.user)
            new_video.save()
            # Redirect to a new page, or add a success message
            messages.success(request, 'Video uploaded successfully.')
            return HttpResponseRedirect(reverse('profile_view'))  
        else:
            # Add an error message
            messages.error(request, 'There was an error uploading your video.')
            return HttpResponseRedirect(reverse('upload_video'))  # assuming 'video_upload' is the name of the URL pattern for the video upload form


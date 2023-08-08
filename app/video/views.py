from rest_framework import generics
from .models import Video, Translation
from .serializers import VideoSerializer, TranslationSerializer
from rest_framework.views import APIView
from rest_framework.response import Response
from .WLASL_Inference.Inference import process_sequence, load_model
import pandas as pd
import subprocess
from django.http import JsonResponse, HttpResponseRedirect
import os
from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from .forms import VideoForm
from django.urls import reverse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.mixins import LoginRequiredMixin

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

def run_script_words(video_file_path, video_id):
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    script_directory = os.path.join(current_directory, 'WLASL_Inference')
    script_path = os.path.join(script_directory, 'Inference.py')

    # Change the working directory to the location of the script
    os.chdir(script_directory)

    result = subprocess.run(['python', script_path, video_file_path, str(video_id)], capture_output=True, text=True)
    output = result.stdout
    error = result.stderr

    # Change the working directory back to its original location
    os.chdir(current_directory)

    return output, error

def run_script_words_live(request):
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    script_directory = os.path.join(current_directory, 'WLASL_Inference')
    script_path = os.path.join(script_directory, 'Inference_live.py')

    # Change the working directory to the location of the script
    os.chdir(script_directory)

    result = subprocess.run(['python', script_path, video_file_path], capture_output=True, text=True)
    output = result.stdout
    error = result.stderr

    # Change the working directory back to its original location
    os.chdir(current_directory)

    return output, error

@login_required
def process_video_words(request, video_id):
    # Get the video from the database
    video = Video.objects.get(id=video_id)
    # Make sure the video belongs to the currently logged-in user
    if video.user != request.user:
            messages.error(request, "You do not have permission to process this video.")
            return HttpResponseRedirect(reverse('list_videos'))
    
    if video.words_status == 'COMPLETED' and request.POST.get('reprocess') != "true":
        return JsonResponse({
            'output': video.transcript_words,
            'error': None
        })
    
    # Get the path of the video file
    video_file_path = video.video_file.path

    video.words_status = 'PROCESSING'
    video.save()

    # Run the script on the video
    output, error = run_script_words(video_file_path, video_id)

    if error:
        video.words_status = 'FAILED'
    else:
        video.words_status = 'COMPLETED'
        video.transcript_words = output

    # Save processed video or other details here if required
    video.save()

    response = {
        'output': output,
        'error': error
    }
    return JsonResponse(response)

def run_script_alphabets(video_file_path, video_id):
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    script_directory = os.path.join(current_directory, 'alphabets')
    script_path = os.path.join(script_directory, 'main.py')

    # Change the working directory to the location of the script
    os.chdir(script_directory)

    result = subprocess.run(['python', script_path, video_file_path, str(video_id)], capture_output=True, text=True)
    output = result.stdout
    error = result.stderr

    # Change the working directory back to its original location
    os.chdir(current_directory)

    return output, error

@login_required
def process_video_alphabets(request, video_id):
    # Get the video from the database
    video = Video.objects.get(id=video_id)
    # Make sure the video belongs to the currently logged-in user
    if video.user != request.user:
            messages.error(request, "You do not have permission to process this video.")
            return HttpResponseRedirect(reverse('list_videos'))
    
    if video.alphabets_status == 'COMPLETED' and request.POST.get('reprocess') != "true":
        return JsonResponse({
            'output': video.transcript_alphabets,
            'error': None
        })
    
    # Get the path of the video file
    video_file_path = video.video_file.path

    video.alphabets_status = 'PROCESSING'
    video.save()

    # Run the script on the video
    output, error = run_script_alphabets(video_file_path, video_id)

    # Check if error contains non-critical message
    is_non_critical_error = "INFO: Created TensorFlow Lite XNNPACK delegate for CPU." in error

    # Adjust condition to ignore non-critical errors
    if error and not is_non_critical_error:
        video.alphabets_status = 'FAILED'
    else:
        video.alphabets_status = 'COMPLETED'
        video.transcript_alphabets = output


    # Save processed video or other details here if required
    video.save()

    response = {
        'output': output,
        'error': error
    }
    return JsonResponse(response)

def run_script_alphabets_live(request):
    current_directory = os.path.dirname(os.path.abspath(__file__)) 
    script_directory = os.path.join(current_directory, 'alphabets')
    script_path = os.path.join(script_directory, 'main_live.py')

    # Change the working directory to the location of the script
    os.chdir(script_directory)

    result = subprocess.run(['python', script_path], capture_output=True, text=True)
    output = result.stdout
    error = result.stderr

    # Change the working directory back to its original location
    os.chdir(current_directory)

    response = {
        'output': output,
        'error': error
    }
    return JsonResponse(response)

class VideoUploadView(View):
    def get(self, request):
        form = VideoForm()
        return render(request, 'video/upload.html', {'form': form})

    def post(self, request, *args, **kwargs):
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            new_video = Video(video_file=request.FILES['video_file'], user=request.user)
            new_video.save()
            # Redirect to a new page, or add a success message
            messages.success(request, 'Video uploaded successfully.')
            return HttpResponseRedirect(reverse('list_videos'))  
        else:
            # Add an error message
            messages.error(request, 'There was an error uploading your video.')
            return HttpResponseRedirect(reverse('upload_video'))  # assuming 'video_upload' is the name of the URL pattern for the video upload form


class VideoListView(LoginRequiredMixin, View):
    def get(self, request):
        videos = Video.objects.filter(user=request.user)
        return render(request, 'video/list.html', {'videos': videos})
     
@login_required
def delete_video(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    # Ensure the video belongs to the currently logged-in user
    if video.user != request.user:
        messages.error(request, "You do not have permission to delete this video.")
        return HttpResponseRedirect(reverse('list_videos'))
    video.delete()
    messages.success(request, "Video deleted.")
    return HttpResponseRedirect(reverse('list_videos'))

@login_required
def view_transcript_words(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    
    # Ensure the logged-in user owns this video
    if video.user != request.user:
        messages.error(request, "You do not have permission to view this transcript.")
        return HttpResponseRedirect(reverse('list_videos'))

    return render(request, 'video/transcript_words.html', {'video': video})

@login_required
def view_transcript_alphabets(request, video_id):
    video = get_object_or_404(Video, id=video_id)
    
    # Ensure the logged-in user owns this video
    if video.user != request.user:
        messages.error(request, "You do not have permission to view this transcript.")
        return HttpResponseRedirect(reverse('list_videos'))

    return render(request, 'video/transcript_alphabets.html', {'video': video})
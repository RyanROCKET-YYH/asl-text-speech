from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
import base64
import os
import json

image_counter = 0

@csrf_exempt
def receive_image(request):
    global image_counter
    if request.method == 'OPTIONS':  
        response = HttpResponse(status=200)
        response["Access-Control-Allow-Origin"] = "*"
        response["Access-Control-Allow-Methods"] = "POST"
        return HttpResponse(status=200)
    elif request.method == 'POST':
        data = request.body
        json_data = json.loads(data)
        img_data = json_data.get('image')

        filename = f'image_{image_counter}.png'
        with open(filename, 'wb') as f:
            f.write(base64.b64decode(img_data))

        image_counter = (image_counter + 1) % 30  #  change 30 into the number of frames need to be saved

        return JsonResponse({"message": "Image received."}, status=200)
    else:
        if image_counter > 0:
            return JsonResponse({"message": "Image received."}, status=200)
        else:
            return JsonResponse({"message": "Waiting for images."}, status=200)


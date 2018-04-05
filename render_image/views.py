from django.shortcuts import render
import scipy.misc as sic
import numpy as np
from PIL import Image
from django.http import JsonResponse
from .inf_temp import index_gan
import time
from django.http import HttpResponse
# Create your views here.
def index(request):
	print(request)
	if request.method == 'POST':
		print(request.FILES)
		im = np.array(Image.open(request.FILES['file2']).convert('RGB'))
	args = {'var': 'cool'}
	return render(request, 'render_image/home.html', args)

def upload_img(request):
	start_time = time.time()
	print(request.FILES)
	size = 600, 600
	im = Image.open(request.FILES['file2']).convert('RGB')
	if im.size > (400, 400):
		im.thumbnail(size, Image.ANTIALIAS)
	im = np.array(im)
	
	# print(type(index_gan(im)))
	im = index_gan(im)
	# print(im.shape)
	with open("test.png", "wb") as f:
		f.write(im)
	print("--- %s seconds ---" % (time.time() - start_time))
	return JsonResponse({'student':'student'})

def render_image(request):
	image_data = open("test.png", "rb").read()
	return HttpResponse(image_data, content_type="image/png")

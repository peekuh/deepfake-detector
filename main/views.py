from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import UserRegisterForm, ImageUploadForm
from .models import UploadedImage

def home(request):
    return render(request, 'home.html')

def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}! You can now log in.')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'register.html', {'form': form})

@login_required
def dashboard(request):
    user_images = UploadedImage.objects.filter(user=request.user).order_by('-uploaded_at')
    return render(request, 'dashboard.html', {'images': user_images})

@login_required
def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image = form.save(commit=False)
            image.user = request.user
            image.save()
            messages.success(request, 'Image uploaded successfully!')
            return redirect('dashboard')
    else:
        form = ImageUploadForm()
    return render(request, 'upload.html', {'form': form})


@login_required
def delete_image(request, image_id):
    """
    Deletes an image uploaded by the user.
    """
    image = get_object_or_404(UploadedImage, pk=image_id, user=request.user)  # Get the image or return a 404 if not found

    if request.method == 'POST':  # Only delete if the request is a POST request
        image.delete()
        messages.success(request, 'Image deleted successfully!')
        return redirect('dashboard')  # Redirect back to the dashboard

    # If it's a GET request, you might want to show a confirmation page
    # (optional, but good practice)
    return render(request, 'confirm_delete.html', {'image': image})  # Create a template named confirm_delete.html

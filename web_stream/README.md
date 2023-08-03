# For BE

## package

pip install django-cors-headers

## settings.py

add 'corsheaders' to INSTALLED_APPS

add 'corsheaders.middleware.CorsMiddleware' to MIDDLEWARE

CORS_ALLOW_ALL_ORIGINS = True

# For FE

line 43: let response = await fetch('change the url into backend url of receive_frame', {
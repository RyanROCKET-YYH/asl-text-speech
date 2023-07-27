from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from crispy_forms.helper import FormHelper

class SignupForm(UserCreationForm):
    pass

class LoginForm(AuthenticationForm):
    pass

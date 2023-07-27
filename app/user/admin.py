from django.contrib import admin

# Register your models here.
from django.contrib.auth.admin import UserAdmin as DefaultUserAdmin
from django.contrib.auth.models import User
from .models import UserProfile

class UserAdmin(DefaultUserAdmin):
    # If you need to modify the admin behavior, you can do it here.
    # For example, you might want to list the email in the admin:
    list_display = DefaultUserAdmin.list_display + ('email',)

admin.site.unregister(User)
admin.site.register(User, UserAdmin)
admin.site.register(UserProfile)
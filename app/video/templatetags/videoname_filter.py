from django import template
import os

register = template.Library()

@register.filter(name='basename')
def basename(value):
    return os.path.basename(value)

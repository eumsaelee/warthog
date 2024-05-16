from django.urls import path
from django.contrib.auth import views as auth_views

from . import views


app_name = 'meerkat'

urlpatterns = [
    path('', views.home, name='home'),
    path('stream/<int:tag>/', views.stream, name='stream')
]
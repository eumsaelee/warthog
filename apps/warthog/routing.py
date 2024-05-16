from django.urls import path
from . import consumers

websocket_urlpatterns = [
    path('ws/stream/<int:tag>/', consumers.Consumer.as_asgi()),
]
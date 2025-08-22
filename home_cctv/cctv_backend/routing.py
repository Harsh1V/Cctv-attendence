from django.urls import re_path
from core.consumers import FaceStreamConsumer

websocket_urlpatterns = [
    re_path(r'ws/face/$', FaceStreamConsumer.as_asgi()),
]

#Ye WebSocket URL /ws/face-stream/ ko hum FaceStreamConsumer se link kar rahe hain.
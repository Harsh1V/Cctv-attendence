import os

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'cctv_backend.settings')

from django.core.asgi import get_asgi_application
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack
import cctv_backend.routing


application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        URLRouter(
            cctv_backend.routing.websocket_urlpatterns  # ✅ hook the routes
        )
    ),
})


# Isse Django ko pata chalega ki WebSocket connection bhi handle karne hain.
import cv2
from django.shortcuts import render
from django.http import StreamingHttpResponse

from .consumers import Consumer


def home(request):
    return render(request, 'home.html')


async def stream(request, tag: int):
    consumer = Consumer.consumers[tag]

    def to_multipart(bytes_frame):
        return (b'--jpeg\r\n'
                b'Content-Type: image/jpeg\r\n\r\n'
                + bytes_frame + b'\r\n')

    async def gen_image():
        while True:
            frame = await consumer._queue.get()
            _, bytes_frame = cv2.imencode('.jpeg', frame)
            yield to_multipart(bytes_frame.tobytes())

    return StreamingHttpResponse(
        gen_image(),
        content_type="multipart/x-mixed-replace; boundary=jpeg"
    )
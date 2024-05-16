import asyncio

import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer

from .src import payload
from .src.buffers import AsyncFrameBuffer


def deserialize(bytes_data: bytes):
    data = payload.Payload()
    data.ParseFromString(bytes_data)
    frame = data.frame
    frame = np.frombuffer(frame, dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_ANYCOLOR)
    preds = {}
    for name, array in data.preds.items():
        data = np.array(array.data, dtype=np.float64)
        data = data.reshape(tuple(array.shape))
        preds[name] = data
    return frame, preds


class Consumer(AsyncWebsocketConsumer):
    consumers = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._tag = None
        self._queue = AsyncFrameBuffer(size=1)
        #self._queue = Queue(maxsize=1)
        self._finished = asyncio.Event()

    async def connect(self):
        self._tag = self.scope['url_route']['kwargs']['tag']
        if self._tag in Consumer.consumers:
            await self.close(code=4001)  # duplicated tag
        Consumer.consumers[self._tag] = self
        await self.accept()

    async def disconnect(self, code):
        await self._queue.put(None)
        await self._finished.wait()
        del Consumer.consumers[self._tag]

    async def set_finished(self):
        self._finished.set()

    async def receive(self, text_data=None, bytes_data=None):
        if text_data:
            pass
        if bytes_data:
            frame, _ = deserialize(bytes_data)
            await self._queue.put(frame)
            #if self._queue.full():
            #    self._queue.get()
            #self._queue.put(frame)
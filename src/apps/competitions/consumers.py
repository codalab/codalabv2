import json
import os
import re

import aiofiles
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings

from competitions.models import Submission


class SubmissionIOConsumer(AsyncWebsocketConsumer):
    #
    async def connect(self):
        submission_id = self.scope['url_route']['kwargs']['submission_id']
        secret = self.scope['url_route']['kwargs']['secret']
        try:
            submission = Submission.objects.get(pk=submission_id, secret=secret)
        except Submission.DoesNotExist:
            return await self.close()

        await self.accept()

    async def receive(self, text_data=None, bytes_data=None):
        # kind = json.loads(text_data).get('kind')

        user_pk = self.scope['url_route']['kwargs']['user_pk']
        submission_id = self.scope['url_route']['kwargs']['submission_id']
        # if kind != 'status_update':
        submission_output_path = os.path.join(settings.TEMP_SUBMISSION_STORAGE, f"{submission_id}.txt")
        os.makedirs(os.path.dirname(submission_output_path), exist_ok=True)

        async with aiofiles.open(submission_output_path, 'a+') as f:
            await f.write(f'{text_data}\n')

        # TODO: Await to broadcast to everyone listening to this submission key or whatever
        await self.channel_layer.group_send(f"submission_listening_{user_pk}", {
            'type': 'submission.message',
            'text': text_data,
            'submission_id': submission_id,
        })
        # TODO! Refuse to write to file after 10MB has been received ???


class SubmissionOutputConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        if not self.scope["user"].is_authenticated:
            return await self.close()

        await self.accept()
        await self.channel_layer.group_add(f"submission_listening_{self.scope['user'].pk}", self.channel_name)

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(f"submission_listening_{self.scope['user'].pk}", self.channel_name)
        await self.close()

    def group_send(self, text, submission_id):
        return self.channel_layer.group_send(f"submission_listening_{self.scope['user'].pk}", {
            'type': 'submission.message',
            'text': text,
            'submission_id': submission_id,
            'full_text': True,
        })

    async def receive(self, text_data=None, bytes_data=None):
        """We expect to receive a message at this endpoint containing the ID of a submission"""
        # Todo: authenticate user has access to submission given the user sent with self.scope['user']
        data = json.loads(text_data)

        print("ws received summin")
        print(data)

        for id in data.get("submission_ids", []):
            text_path = os.path.join(settings.TEMP_SUBMISSION_STORAGE, f"{id}.txt")
            if os.path.exists(text_path):
                with open(text_path) as f:
                    text = f.read()
                await self.group_send(text, id)

        # submission_id = text_data
        # text_path = os.path.join(settings.TEMP_SUBMISSION_STORAGE, f"{submission_id}.txt")
        # if os.path.exists(text_path):
        #     with open(text_path) as f:
        #         text = f.read()
        #     await self.group_send(text, submission_id)
        #     # TODO: fix potential security issue? get other peoples submission logs on page refresh
        #     #  if code submission has child id print statements
        #     # TODO this feels weird, we could make this all a bit cleaner
        #     children = re.findall(r'child_id\": (\d+)', text)
        #     for child_id in children:
        #         child_text_path = os.path.join(settings.TEMP_SUBMISSION_STORAGE, f"{child_id}.txt")
        #         if os.path.exists(child_text_path):
        #             with open(child_text_path) as f:
        #                 child_text = f.read()
        #             await self.group_send(child_text, child_id)

    async def submission_message(self, event):
        data = {
            "type": "catchup" if event.get('full_text') else "message",
            "submission_id": event['submission_id'],
            "data": event['text']
        }
        await self.send(json.dumps(data))

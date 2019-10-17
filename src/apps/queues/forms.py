from django.core.exceptions import ValidationError
from django.forms import ModelForm

from queues.models import Queue


class QueueForm(ModelForm):

    class Meta:
        model = Queue
        fields = ('name', 'is_public', 'organizers')

    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user')
        super().__init__(*args, **kwargs)

        # Remove extra help text that obfuscates our organizer message
        remove_message = 'Hold down "Control", or "Command" on a Mac, to select more than one.'
        self.fields['organizers'].help_text = self.fields['organizers'].help_text.replace(remove_message, '')

    def clean(self):
        # If we're creating this, make sure we don't have > the limited # of queues
        if not self.instance.pk:
            if self.user.queues.count() >= self.user.rabbitmq_queue_limit:
                raise ValidationError(f"Cannot create more than {self.user.rabbitmq_queue_limit} queues")
        return super().clean()

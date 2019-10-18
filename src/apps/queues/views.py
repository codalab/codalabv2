from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.exceptions import PermissionDenied
from django.forms.utils import ErrorList
from django.http import HttpResponseRedirect
from django.urls import reverse
from django.views.generic import ListView, CreateView, UpdateView, DeleteView
from pyrabbit.http import HTTPError
from queues.forms import QueueForm
from queues.models import Queue

from . import rabbit


class QueueFormMixin:
    # Necessary for DeleteQueueView, base class references missing `self.object`
    def get_success_url(self):
        return reverse('queues:list')

    def get_form_kwargs(self, *args, **kwargs):
        kwargs = super().get_form_kwargs(*args, **kwargs)
        kwargs['user'] = self.request.user
        return kwargs


class QueueListView(LoginRequiredMixin, ListView):
    model = Queue
    template_name = 'queues/list.html'

    def get_queryset(self):
        if self.request.user.is_superuser:
            return Queue.objects.all()
        else:
            return self.request.user.queues.all() | self.request.user.organized_queues.all()


class QueueCreateView(LoginRequiredMixin, QueueFormMixin, CreateView):
    model = Queue
    template_name = 'queues/form.html'
    form_class = QueueForm

    def form_valid(self, form):
        try:
            queue = form.save(commit=False)
            queue.owner = self.request.user
            queue.vhost = rabbit.create_queue(self.request.user)
            # Only save queue if things were successful
            queue.save()
            # Save Many2Many fields
            form.save_m2m()
            return HttpResponseRedirect(self.get_success_url())
        except HTTPError:
            # To inject additional non-field errors
            errors = form._errors.setdefault("__all__", ErrorList())
            errors.append("Failed to create RabbitMQ queue... please report this issue on the Codalab github!")
            return self.form_invalid(form)


class QueueUpdateView(LoginRequiredMixin, QueueFormMixin, UpdateView):
    model = Queue
    template_name = 'queues/form.html'
    form_class = QueueForm


class QueueDeleteView(LoginRequiredMixin, QueueFormMixin, DeleteView):
    model = Queue
    template_name = 'queues/delete.html'

    def delete(self, request, *args, **kwargs):
        queue = self.get_object()
        if request.user != queue.owner:
            raise PermissionDenied("Cannot delete this queue, you don't own it.")
        success_url = self.get_success_url()
        queue.delete()
        return HttpResponseRedirect(success_url)

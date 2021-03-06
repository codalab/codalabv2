# Generated by Django 2.2.13 on 2020-09-21 18:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tasks', '0003_task_order'),
        ('competitions', '0019_remove_phase_tasks'),
    ]

    operations = [
        migrations.AddField(
            model_name='phase',
            name='tasks',
            field=models.ManyToManyField(blank=True, related_name='phases', through='competitions.PhaseTaskInstance', to='tasks.Task'),
        ),
    ]

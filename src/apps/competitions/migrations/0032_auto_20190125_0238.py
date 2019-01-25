# Generated by Django 2.1.2 on 2019-01-25 02:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('competitions', '0031_phase_status'),
    ]

    operations = [
        migrations.AddField(
            model_name='competitionparticipant',
            name='reason',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
        migrations.AddField(
            model_name='competitionparticipant',
            name='status',
            field=models.CharField(choices=[('unknown', 'unknown'), ('denied', 'denied'), ('approved', 'approved'), ('pending', 'pending')], default='unknown', max_length=128),
        ),
    ]

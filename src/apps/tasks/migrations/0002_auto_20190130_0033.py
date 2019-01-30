# Generated by Django 2.1.2 on 2019-01-30 00:33

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('tasks', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='ingestionmodule',
            name='created_when',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='scoringmodule',
            name='created_when',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
        migrations.AlterField(
            model_name='task',
            name='created_when',
            field=models.DateTimeField(default=django.utils.timezone.now),
        ),
    ]

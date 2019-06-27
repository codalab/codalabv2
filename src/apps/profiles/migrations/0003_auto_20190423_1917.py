# Generated by Django 2.1.2 on 2019-04-23 19:17

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('profiles', '0002_auto_20190214_1845'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='chahub_data_hash',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='user',
            name='chahub_needs_retry',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='user',
            name='chahub_timestamp',
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]

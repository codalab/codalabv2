# Generated by Django 2.0.1 on 2018-02-07 01:58

from django.db import migrations, models
import src.utils.data


class Migration(migrations.Migration):

    dependencies = [
        ('competitions', '0009_auto_20180207_0056'),
    ]

    operations = [
        migrations.AddField(
            model_name='submission',
            name='description',
            field=models.CharField(blank=True, default='', max_length=120, null=True),
        ),
        migrations.AddField(
            model_name='submission',
            name='name',
            field=models.CharField(blank=True, default='', max_length=120, null=True),
        ),
        migrations.AlterField(
            model_name='submission',
            name='zip_file',
            field=models.FileField(blank=True, null=True, upload_to=src.utils.data.PathWrapper('submissions')),
        ),
    ]

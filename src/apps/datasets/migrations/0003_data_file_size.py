# Generated by Django 2.1.2 on 2019-10-09 19:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('datasets', '0002_auto_20190925_2256'),
    ]

    operations = [
        migrations.AddField(
            model_name='data',
            name='file_size',
            field=models.PositiveIntegerField(blank=True, null=True),
        ),
    ]

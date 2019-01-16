# Generated by Django 2.1.2 on 2019-01-16 20:44

from django.db import migrations, models
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('datasets', '0006_auto_20190110_2157'),
    ]

    operations = [
        migrations.AlterField(
            model_name='data',
            name='key',
            field=models.UUIDField(blank=True, default=uuid.uuid4, unique=True),
        ),
    ]

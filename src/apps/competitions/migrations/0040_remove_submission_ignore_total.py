# Generated by Django 2.1.2 on 2019-07-01 21:53

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('competitions', '0039_merge_20190701_1837'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='submission',
            name='ignore_total',
        ),
    ]

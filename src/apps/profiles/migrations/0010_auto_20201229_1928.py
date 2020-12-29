# Generated by Django 2.2.13 on 2020-12-29 19:28

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('profiles', '0009_auto_20201219_1308'),
    ]

    operations = [
        migrations.AlterField(
            model_name='user',
            name='personal_url',
            field=models.URLField(blank=True, error_messages={'invalid': 'Enter a valid URL. Hint: URL must start with `https://` or `http://`'}, null=True),
        ),
    ]

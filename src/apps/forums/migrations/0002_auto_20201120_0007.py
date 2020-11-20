# Generated by Django 2.2.13 on 2020-11-20 00:07

from django.db import migrations


def create_forums_for_competitions(apps, schema_editor):
    Competition = apps.get_model('competitions', 'Competition')
    Forum = apps.get_model('forums', 'Forum')
    for competition in Competition.objects.all():
        Forum.objects.create(competition=competition)


class Migration(migrations.Migration):

    dependencies = [
        ('forums', '0001_initial'),
    ]

    operations = [
        migrations.RunPython(create_forums_for_competitions),
    ]

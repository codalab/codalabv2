# Generated by Django 2.1.2 on 2019-06-12 20:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('competitions', '0034_competition_has_been_migrated'),
    ]

    operations = [
        migrations.RenameField(
            model_name='phase',
            old_name='auto_migration',
            new_name='auto_migrate_to_this_phase',
        ),
        migrations.RemoveField(
            model_name='competition',
            name='has_been_migrated',
        ),
        migrations.AddField(
            model_name='phase',
            name='has_been_migrated',
            field=models.BooleanField(default=False),
        ),
    ]
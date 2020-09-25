# Generated by Django 2.2.13 on 2020-09-21 17:43

from django.db import migrations


def create_through_relations(apps, schema_editor):
    Phase = apps.get_model('competitions', 'Phase')
    TaskOrder = apps.get_model('competitions', 'TaskOrder')
    for phase in Phase.objects.all():
        for task in phase.tasks.all():
            TaskOrder(
                phase=phase,
                task=task,
                order_index=999,
            ).save()


class Migration(migrations.Migration):
    dependencies = [
        ('competitions', '0017_taskorder'),
    ]

    operations = [
        migrations.RunPython(create_through_relations, reverse_code=migrations.RunPython.noop),
    ]

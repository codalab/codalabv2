# Generated by Django 2.1.2 on 2019-01-04 23:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('competitions', '0015_auto_20180820_2329'),
    ]

    operations = [
        migrations.AddField(
            model_name='phase',
            name='status',
            field=models.TextField(blank=True, choices=[('Previous', 'Previous'), ('Current', 'Current'), ('Next', 'Next'), ('Final', 'Final')], null=True),
        ),
    ]

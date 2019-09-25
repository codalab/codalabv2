# Generated by Django 2.1.2 on 2019-09-25 22:56

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('datasets', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Solution',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
                ('description', models.TextField(blank=True, null=True)),
                ('key', models.UUIDField(blank=True, default=uuid.uuid4, unique=True)),
                ('data', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='datasets.Data')),
            ],
        ),
        migrations.CreateModel(
            name='Task',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=256)),
                ('description', models.TextField(blank=True, null=True)),
                ('key', models.UUIDField(blank=True, default=uuid.uuid4, unique=True)),
                ('created_when', models.DateTimeField(default=django.utils.timezone.now)),
                ('is_public', models.BooleanField(default=False)),
                ('ingestion_only_during_scoring', models.BooleanField(default=False)),
                ('created_by', models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to=settings.AUTH_USER_MODEL)),
                ('ingestion_program', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='task_ingestion_programs', to='datasets.Data')),
                ('input_data', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='task_input_datas', to='datasets.Data')),
                ('reference_data', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='task_reference_datas', to='datasets.Data')),
                ('scoring_program', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='task_scoring_programs', to='datasets.Data')),
            ],
        ),
        migrations.AddField(
            model_name='solution',
            name='tasks',
            field=models.ManyToManyField(related_name='solutions', to='tasks.Task'),
        ),
    ]

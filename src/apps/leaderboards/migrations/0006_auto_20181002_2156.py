# Generated by Django 2.1 on 2018-10-02 21:56

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('competitions', '0017_auto_20181002_2156'),
        ('leaderboards', '0005_auto_20180814_1851'),
    ]

    operations = [
        migrations.CreateModel(
            name='SubmissionScore',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('score', models.DecimalField(decimal_places=10, max_digits=20)),
                ('column', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='leaderboards.Column')),
                ('submission', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='scores', to='competitions.Submission')),
            ],
        ),
        migrations.RemoveField(
            model_name='submissionresult',
            name='column',
        ),
        migrations.RemoveField(
            model_name='submissionresult',
            name='submission',
        ),
        migrations.DeleteModel(
            name='SubmissionResult',
        ),
    ]

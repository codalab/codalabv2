# Generated by Django 2.1.2 on 2019-09-25 22:56

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('tasks', '0001_initial'),
        ('datasets', '0001_initial'),
        ('leaderboards', '0001_initial'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('competitions', '0002_auto_20190925_2256'),
    ]

    operations = [
        migrations.AddField(
            model_name='submission',
            name='owner',
            field=models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, related_name='submission', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='submission',
            name='parent',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='children', to='competitions.Submission'),
        ),
        migrations.AddField(
            model_name='submission',
            name='participant',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='submissions', to='competitions.CompetitionParticipant'),
        ),
        migrations.AddField(
            model_name='submission',
            name='phase',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='submissions', to='competitions.Phase'),
        ),
        migrations.AddField(
            model_name='submission',
            name='scores',
            field=models.ManyToManyField(related_name='submissions', to='leaderboards.SubmissionScore'),
        ),
        migrations.AddField(
            model_name='phase',
            name='competition',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='phases', to='competitions.Competition'),
        ),
        migrations.AddField(
            model_name='phase',
            name='tasks',
            field=models.ManyToManyField(blank=True, related_name='phases', to='tasks.Task'),
        ),
        migrations.AddField(
            model_name='page',
            name='competition',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='pages', to='competitions.Competition'),
        ),
        migrations.AddField(
            model_name='competitionparticipant',
            name='competition',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='participants', to='competitions.Competition'),
        ),
        migrations.AddField(
            model_name='competitionparticipant',
            name='user',
            field=models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, related_name='competitions_im_in', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='competitiondump',
            name='competition',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='dumps', to='competitions.Competition'),
        ),
        migrations.AddField(
            model_name='competitiondump',
            name='dataset',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='competition_dump_file', to='datasets.Data'),
        ),
        migrations.AddField(
            model_name='competitioncreationtaskstatus',
            name='dataset',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='competition_bundles', to='datasets.Data'),
        ),
        migrations.AddField(
            model_name='competitioncreationtaskstatus',
            name='resulting_competition',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to='competitions.Competition'),
        ),
        migrations.AddField(
            model_name='competition',
            name='collaborators',
            field=models.ManyToManyField(blank=True, related_name='collaborations', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AddField(
            model_name='competition',
            name='created_by',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='competitions', to=settings.AUTH_USER_MODEL),
        ),
        migrations.AlterUniqueTogether(
            name='submission',
            unique_together={('owner', 'leaderboard')},
        ),
    ]

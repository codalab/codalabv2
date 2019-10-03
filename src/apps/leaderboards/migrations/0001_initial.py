# Generated by Django 2.1.2 on 2019-09-25 22:56

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('competitions', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Column',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('computation', models.TextField(blank=True, choices=[('avg', 'Average')], null=True)),
                ('computation_indexes', models.TextField(blank=True, max_length=255, null=True)),
                ('title', models.CharField(max_length=36)),
                ('key', models.CharField(max_length=36)),
                ('sorting', models.TextField(choices=[('desc', 'Descending'), ('asc', 'Ascending')], default='desc')),
                ('index', models.PositiveIntegerField()),
            ],
            options={
                'ordering': ('index',),
            },
        ),
        migrations.CreateModel(
            name='Leaderboard',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('primary_index', models.PositiveIntegerField(default=0)),
                ('title', models.CharField(max_length=64)),
                ('key', models.CharField(max_length=36)),
                ('competition', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='leaderboards', to='competitions.Competition')),
            ],
        ),
        migrations.CreateModel(
            name='SubmissionScore',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('score', models.DecimalField(decimal_places=10, max_digits=20)),
                ('column', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='leaderboards.Column')),
            ],
        ),
        migrations.AddField(
            model_name='column',
            name='leaderboard',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='columns', to='leaderboards.Leaderboard'),
        ),
        migrations.AlterUniqueTogether(
            name='column',
            unique_together={('leaderboard', 'key')},
        ),
    ]

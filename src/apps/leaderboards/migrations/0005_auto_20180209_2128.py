# Generated by Django 2.0.2 on 2018-02-09 21:28

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('leaderboards', '0004_auto_20180126_1800'),
    ]

    operations = [
        migrations.AlterField(
            model_name='leaderboard',
            name='competition',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='leaderboards', to='competitions.Competition'),
        ),
    ]

import asyncio
from os.path import basename

from django.core.exceptions import ValidationError
from rest_framework import serializers

from api.serializers import leaderboards
from competitions.models import Submission, SubmissionDetails
from datasets.models import Data
from leaderboards.models import SubmissionScore
from utils.data import make_url_sassy


class SubmissionScoreSerializer(serializers.ModelSerializer):
    index = serializers.IntegerField(source='column.index', read_only=True)
    column_key = serializers.CharField(source='column.key', read_only=True)

    class Meta:
        model = SubmissionScore
        fields = (
            'id',
            'index',
            'score',
            'column_key',
        )


class SubmissionSerializer(serializers.ModelSerializer):
    scores = SubmissionScoreSerializer(many=True)
    filename = serializers.SerializerMethodField(read_only=True)
    owner = serializers.CharField(source='owner.username')
    phase_name = serializers.CharField(source='phase.name')

    class Meta:
        model = Submission
        fields = (
            'phase',
            'phase_name',
            'name',
            'filename',
            'description',
            'pk',
            'id',
            'created_when',
            'is_public',
            'status',
            'status_details',
            'scores',
            'leaderboard',
            'owner',
            'has_children',
            'parent',
            'children',
        )
        extra_kwargs = {
            "phase": {"read_only": True},
            "scores": {"read_only": True},
            "leaderboard": {"read_only": True},
        }

    def get_filename(self, instance):
        return basename(instance.data.data_file.name)


class SubmissionLeaderBoardSerializer(serializers.ModelSerializer):
    scores = SubmissionScoreSerializer(many=True)
    owner = serializers.CharField(source='owner.username')

    class Meta:
        model = Submission
        fields = (
            'scores',
            'owner',
        )
        extra_kwargs = {
            "scores": {"read_only": True},
            "owner": {"read_only": True},
        }


class SubmissionCreationSerializer(serializers.ModelSerializer):
    data = serializers.SlugRelatedField(queryset=Data.objects.all(), required=False, allow_null=True, slug_field='key')
    filename = serializers.SerializerMethodField(read_only=True)

    class Meta:
        model = Submission
        fields = (
            'id',
            'data',
            'phase',
            'status',
            'status_details',
            'filename',
            'description',
            'secret',
        )
        extra_kwargs = {
            'secret': {"write_only": True},
            'description': {"read_only": True},
            # 'status': {"read_only": True},
        }

    def get_filename(self, instance):
        return basename(instance.data.data_file.name)

    # TODO: Validate the user is a participant in this competition.phase

    def create(self, validated_data):
        validated_data["owner"] = self.context['owner']
        sub = super().create(validated_data)
        sub.start()
        return sub

    def validate(self, attrs):
        data = super().validate(attrs)
        task_pk = self._kwargs.get('data', {}).get('task_pk')
        if task_pk:
            data['task_pk'] = task_pk
        return data

    def update(self, instance, validated_data):
        if instance.secret != validated_data.get('secret'):
            raise PermissionError("Submission secret invalid")
        print("Updated to...")
        print(validated_data)

        if "status" in validated_data:
            # Received a status update, let the frontend know
            from channels.layers import get_channel_layer
            channel_layer = get_channel_layer()
            asyncio.get_event_loop().run_until_complete(channel_layer.group_send(f"submission_listening_{instance.owner.pk}", {
                'type': 'submission.message',
                'text': {
                    "kind": "status_update",
                    "status": validated_data["status"],
                },
                'submission_id': instance.id,
                'full_text': True,
            }))

        if validated_data["status"] == Submission.SCORING:
            # Start scoring because we're "SCORING" status now from compute worker
            from competitions.tasks import run_submission
            task_id = validated_data.get('task_pk')
            if not task_id:
                raise ValidationError('Cannot update submission. Task pk was not provided')
            run_submission(instance.pk, task_pk=task_id, is_scoring=True)
        resp = super().update(instance, validated_data)
        if instance.parent:
            instance.parent.check_child_submission_statuses()
        return resp


class SubmissionDetailSerializer(serializers.ModelSerializer):
    data_file = serializers.SerializerMethodField()

    class Meta:
        model = SubmissionDetails
        fields = (
            'name',
            'data_file',
        )

    def get_data_file(self, instance):
        return make_url_sassy(instance.data_file.name)


class SubmissionFilesSerializer(serializers.ModelSerializer):
    logs = serializers.SerializerMethodField()
    data_file = serializers.SerializerMethodField()
    prediction_result = serializers.SerializerMethodField()
    scoring_result = serializers.SerializerMethodField()
    leaderboards = serializers.SerializerMethodField()

    class Meta:
        model = Submission
        fields = (
            'logs',
            'data_file',
            'prediction_result',
            'scoring_result',
            'leaderboards',
        )

    def get_logs(self, instance):
        return SubmissionDetailSerializer(instance.details.all(), many=True).data

    def get_data_file(self, instance):
        return make_url_sassy(instance.data.data_file.name)

    def get_prediction_result(self, instance):
        if instance.prediction_result.name:
            return make_url_sassy(instance.prediction_result.name)

    def get_scoring_result(self, instance):
        if instance.scoring_result.name:
            return make_url_sassy(instance.scoring_result.name)

    def get_leaderboards(self, instance):
        boards = list(set([score.column.leaderboard for score in instance.scores.all().select_related('column__leaderboard')]))
        return [leaderboards.LeaderboardSerializer(lb).data for lb in boards]

from drf_writable_nested import WritableNestedModelSerializer
from rest_framework import serializers

from api.fields import NamedBase64ImageField
from api.serializers.leaderboards import LeaderboardSerializer
from api.serializers.profiles import CollaboratorSerializer
from api.serializers.tasks import TaskListSerializer
from competitions.models import Competition, Phase, Page, CompetitionCreationTaskStatus, CompetitionParticipant
from profiles.models import User
from tasks.models import Task


class PhaseSerializer(WritableNestedModelSerializer):
    tasks = serializers.SlugRelatedField(queryset=Task.objects.all(), required=False, allow_null=True, slug_field='key',
                                         many=True)

    class Meta:
        model = Phase
        fields = (
            'id',
            'index',
            'start',
            'end',
            'name',
            'description',
            'status',
            'execution_time_limit',
            'tasks',
            'has_max_submissions',
            'max_submissions_per_day',
            'max_submissions_per_person',
            'auto_migrate_to_this_phase',
        )


class PhaseDetailSerializer(serializers.ModelSerializer):
    tasks = TaskListSerializer(many=True)

    class Meta:
        model = Phase
        fields = (
            'id',
            'index',
            'start',
            'end',
            'name',
            'description',
            'status',
            'tasks',
            'auto_migrate_to_this_phase',
            'has_max_submissions',
            'max_submissions_per_day',
            'max_submissions_per_person',
            'execution_time_limit',
        )


class PageSerializer(WritableNestedModelSerializer):
    # *NOTE* The competition property has to be replicated at the end of the file
    # after the CompetitionSerializer class is declared
    # competition = CompetitionSerializer(many=True)

    class Meta:
        model = Page
        fields = (
            'id',
            'title',
            'content',
            'index',
        )


class CompetitionSerializer(WritableNestedModelSerializer):
    created_by = serializers.SerializerMethodField(read_only=True)
    logo = NamedBase64ImageField(required=True)
    pages = PageSerializer(many=True)
    phases = PhaseSerializer(many=True)
    leaderboards = LeaderboardSerializer(many=True)
    collaborators = serializers.PrimaryKeyRelatedField(queryset=User.objects.all(), many=True, required=False)

    class Meta:
        model = Competition
        fields = (
            'id',
            'title',
            'published',
            'secret_key',
            'created_by',
            'created_when',
            'logo',
            'docker_image',
            'pages',
            'phases',
            'leaderboards',
            'collaborators',
            'description',
            'terms',
            'registration_auto_approve',
        )

    def get_created_by(self, object):
        return str(object.created_by)

    def validate_leaderboards(self, value):
        if not value:
            raise serializers.ValidationError("Competitions require at least 1 leaderboard")
        return value

    def validate_phases(self, phases):
        if not phases or len(phases) <= 0:
            raise serializers.ValidationError("Competitions must have at least one phase")
        if len(phases) == 1 and phases[0].get('auto_migrate_to_this_phase'):
            raise serializers.ValidationError("You cannot auto migrate in a competition with one phase")
        if phases[0].get('auto_migrate_to_this_phase') is True:
            raise serializers.ValidationError("You cannot auto migrate to the first phase of a competition")
        return phases

    def create(self, validated_data):
        validated_data["created_by"] = self.context['created_by']
        return super().create(validated_data)


class CompetitionDetailSerializer(serializers.ModelSerializer):
    created_by = serializers.CharField(source='created_by.username', read_only=True)
    logo = NamedBase64ImageField(required=True)
    pages = PageSerializer(many=True)
    phases = PhaseDetailSerializer(many=True)
    leaderboards = LeaderboardSerializer(many=True)
    collaborators = CollaboratorSerializer(many=True)
    participant_status = serializers.CharField(read_only=True)
    participant_count = serializers.IntegerField(read_only=True)
    submission_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Competition
        fields = (
            'id',
            'title',
            'published',
            'secret_key',
            'created_by',
            'created_when',
            'logo',
            'terms',
            'pages',
            'phases',
            'leaderboards',
            'collaborators',
            'participant_status',
            'registration_auto_approve',
            'description',
            'participant_count',
            'submission_count',
        )


class CompetitionSerializerSimple(serializers.ModelSerializer):
    created_by = serializers.CharField(source='created_by.username')
    participant_count = serializers.CharField(read_only=True)
    logo = NamedBase64ImageField(required=False)

    class Meta:
        model = Competition
        fields = (
            'id',
            'title',
            'created_by',
            'created_when',
            'published',
            'participant_count',
            'logo',
            'description',
        )


PageSerializer.competition = CompetitionSerializer(many=True, source='competition')


class CompetitionCreationTaskStatusSerializer(serializers.ModelSerializer):
    class Meta:
        model = CompetitionCreationTaskStatus
        fields = (
            'status',
            'details',
            'resulting_competition',
        )


class CompetitionParticipantSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source='user.username')
    email = serializers.CharField(source='user.email')

    class Meta:
        model = CompetitionParticipant
        fields = (
            'id',
            'username',
            'email',
            'status',
        )

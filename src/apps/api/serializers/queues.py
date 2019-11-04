from rest_framework import serializers
from queues.models import Queue

from profiles.models import User


class OrganizerSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = (
            'username',
            'email',
            'id',
        )


class QueueCreationSerializer(serializers.ModelSerializer):
    is_owner = serializers.SerializerMethodField()
    organizers = serializers.PrimaryKeyRelatedField(queryset=User.objects.all(), many=True, required=False)

    class Meta:
        model = Queue
        fields = (
            'name',
            'vhost',
            'is_public',
            'owner',
            'organizers',
            'broker_url',
            'created_when',
            'is_owner',
            'id',
        )
        read_only_fields = (
            'broker_url',
            'vhost',
            'created_when',
            'is_owner',
        )

    def get_is_owner(self, instance):
        return instance.owner == self.context.get('owner')


class QueueSerializer(serializers.ModelSerializer):
    is_owner = serializers.SerializerMethodField()
    owner = serializers.CharField(source='owner.username', read_only=True)
    organizers = OrganizerSerializer(many=True, read_only=True)

    class Meta:
        model = Queue
        fields = (
            'name',
            'vhost',
            'is_public',
            'owner',
            'organizers',
            'broker_url',
            'created_when',
            'is_owner',
            'id',
        )
        read_only_fields = (
            'broker_url',
            'vhost',
            'created_when',
            'is_owner',
        )

    def get_is_owner(self, instance):
        request = self.context.get('request')
        if not request:
            return None
        return instance.owner == request.user

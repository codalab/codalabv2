
from unittest import mock
from datetime import timedelta
from django.test import TestCase
from django.utils.timezone import now

from competitions.models import Submission, Competition, Phase
from competitions.tasks import do_phase_migrations
from factories import UserFactory, CompetitionFactory, PhaseFactory, SubmissionFactory, SubmissionScoreFactory, \
    CompetitionParticipantFactory, TaskFactory

twenty_minutes_ago = now() - timedelta(hours=0, minutes=20)
twenty_five_minutes_ago = now() - timedelta(hours=0, minutes=25)
five_minutes_ago = now() - timedelta(hours=0, minutes=5)
twenty_minutes_from_now = now() + timedelta(hours=0, minutes=20)


class PhaseToPhaseMigrationTests(TestCase):
    def setUp(self):
        self.owner = UserFactory(username='owner', super_user=True)
        self.normal_user = UserFactory(username='norm')
        self.competition = CompetitionFactory(created_by=self.owner, title="Competition One")
        self.competition_participant = CompetitionParticipantFactory(user=self.normal_user,
                                                                     competition=self.competition)
        self.phase1 = PhaseFactory(
            competition=self.competition,
            auto_migrate_to_this_phase=False,
            start=twenty_five_minutes_ago,
            end=twenty_minutes_ago,
            index=0,
            name='Phase1',
            status=Phase.CURRENT
        )

        self.phase2 = PhaseFactory(
            competition=self.competition,
            auto_migrate_to_this_phase=True,
            start=five_minutes_ago,
            end=twenty_minutes_from_now,
            index=1,
            name='Phase2',
            status=Phase.NEXT
        )

        self.phase3 = PhaseFactory(
            competition=self.competition,
            auto_migrate_to_this_phase=False,
            start=twenty_minutes_from_now,
            index=2,
            name='Phase3',
            status=Phase.FINAL
        )

        for _ in range(4):
            self.make_submission()

    def make_submission(self, **kwargs):
        kwargs.setdefault('owner', self.owner)
        kwargs.setdefault('participant', self.competition_participant)
        kwargs.setdefault('phase', self.phase1)
        kwargs.setdefault('status', 'None')
        sub = SubmissionFactory(**kwargs)
        return sub

    def mock_migration(self):
        with mock.patch('competitions.models.Submission.start') as submission_start:
            do_phase_migrations()
            return submission_start

    def test_migrate_submissions(self):
        assert not self.phase2.submissions.exists()
        mock_start = self.mock_migration()
        assert mock_start.call_count == self.phase1.submissions.count()
        assert self.phase1.submissions.count() == self.phase2.submissions.count()

    def test_currently_migrating_competitions_dont_migrate(self):
        self.mock_migration()
        assert Competition.objects.get(id=self.competition.id).is_migrating
        mock_start = self.mock_migration()
        assert not mock_start.called

    def test_competitions_with_scoring_submissions_dont_migrate(self):
        self.make_submission(status=Submission.SCORING, participant=self.competition_participant)
        self.mock_migration()
        assert self.phase1.submissions.count() != self.phase2.submissions.count()

    def test_submission_ran_after_migration_complete(self):
        self.mock_migration()
        assert not self.phase2.submissions.filter(status='None').exists()

    def test_has_been_migrated_competitions_arent_migrated(self):
        self.phase2.has_been_migrated = True
        self.phase2.save()
        assert not self.phase2.submissions.exists()
        self.mock_migration()
        assert self.phase1.submissions.count() != self.phase2.submissions.count()

    def test_prevent_migration_to_auto_migrate_to_the_phase_is_false(self):
        assert not self.phase2.submissions.exists()
        assert not self.phase3.submissions.exists()
        self.mock_migration()
        assert self.phase1.submissions.count() == self.phase2.submissions.count()
        assert self.phase1.submissions.count() != self.phase3.submissions.count()
        assert self.phase2.submissions.count() != self.phase3.submissions.count()

    def test_all_submissions_migrated_before_changing_phase_status(self):
        self.mock_migration()
        assert Phase.objects.get(id=self.phase1.id).status == Phase.PREVIOUS
        assert Competition.objects.get(id=self.competition.id).is_migrating
        phase2 = Phase.objects.get(id=self.phase2.id)
        assert phase2.has_been_migrated
        assert phase2.status == Phase.CURRENT

        self.mock_migration()
        assert Competition.objects.get(id=self.competition.id).is_migrating

        self.phase2.submissions.update(status='Finished')
        self.mock_migration()
        assert not Competition.objects.get(id=self.phase1.competition.id).is_migrating

        mock_start = self.mock_migration()
        assert mock_start.call_count == 0


class PhaseStatusTests(TestCase):
    def setUp(self):
        self.user = UserFactory()
        self.comp = CompetitionFactory(created_by=self.user)
        self.tasks = [TaskFactory(created_by=self.user)]
        base = {'competition': self.comp, 'tasks': self.tasks}
        self.phase1 = PhaseFactory(
            **base,
            start=now() - timedelta(minutes=10),
            end=now() - timedelta(minutes=5),
        )
        self.phase2 = PhaseFactory(
            **base,
            start=now() - timedelta(minutes=1),
            end=now() + timedelta(minutes=5),
        )
        self.phase3 = PhaseFactory(
            **base,
            start=now() + timedelta(minutes=10),
            end=now() + timedelta(minutes=15),
        )

    def test_phase_migration_updates_statuses(self):
        do_phase_migrations()
        assert Phase.objects.get(id=self.phase1.id).status == Phase.PREVIOUS
        assert Phase.objects.get(id=self.phase2.id).status == Phase.CURRENT
        assert Phase.objects.get(id=self.phase3.id).status == Phase.NEXT

        self.phase2.start = now() - timedelta(minutes=5)
        self.phase2.end = now() - timedelta(minutes=3)
        self.phase2.save(update_fields=['start', 'end'])
        do_phase_migrations()
        assert Phase.objects.get(id=self.phase1.id).status is None
        assert Phase.objects.get(id=self.phase2.id).status == Phase.PREVIOUS
        assert Phase.objects.get(id=self.phase3.id).status == Phase.NEXT

        self.phase3.start = now() - timedelta(minutes=2)
        self.phase3.end = now() + timedelta(minutes=3)
        self.phase3.save(update_fields=['start', 'end'])
        do_phase_migrations()
        assert Phase.objects.get(id=self.phase1.id).status is None
        assert Phase.objects.get(id=self.phase2.id).status == Phase.PREVIOUS
        assert Phase.objects.get(id=self.phase3.id).status == Phase.CURRENT

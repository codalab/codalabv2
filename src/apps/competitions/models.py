from django.conf import settings
from django.db import models

from utils.data import PathWrapper


class Competition(models.Model):
    title = models.CharField(max_length=256)
    logo = models.ImageField(upload_to=PathWrapper('logos'), null=True, blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="competitions")
    created_when = models.DateTimeField(auto_now_add=True)
    collaborators = models.ManyToManyField(settings.AUTH_USER_MODEL, related_name="collaborations", blank=True)

    def __str__(self):
        return "competition-{0}-{1}".format(self.title, self.pk)


class CompetitionCreationTaskStatus(models.Model):
    STARTING = "Starting"
    FINISHED = "Finished"
    FAILED = "Failed"

    STATUS_CHOICES = (
        (STARTING, "None"),
        (FINISHED, "Finished"),
        (FAILED, "Failed"),
    )

    dataset = models.ForeignKey('datasets.Data', on_delete=models.CASCADE, related_name="competition_bundles")
    status = models.TextField(choices=STATUS_CHOICES, null=True, blank=True)
    details = models.TextField(null=True, blank=True)

    # The resulting competition is only made on success
    resulting_competition = models.ForeignKey(Competition, on_delete=models.CASCADE, null=True, blank=True)

    def __str__(self):
        return f"Comp uploaded by {self.dataset.created_by} - {self.status}"


class Phase(models.Model):
    competition = models.ForeignKey(Competition, on_delete=models.CASCADE, null=True, blank=True, related_name='phases')
    index = models.PositiveIntegerField()
    start = models.DateTimeField()
    end = models.DateTimeField(null=True, blank=True)
    name = models.CharField(max_length=256)
    description = models.TextField(null=True, blank=True)

    # These related names are all garbage. Had to do it this way just to prevent clashes...
    input_data = models.ForeignKey('datasets.Data', on_delete=models.SET_NULL, null=True, blank=True, related_name="input_datas")
    reference_data = models.ForeignKey('datasets.Data', on_delete=models.SET_NULL, null=True, blank=True, related_name="reference_datas")
    scoring_program = models.ForeignKey('datasets.Data', on_delete=models.SET_NULL, null=True, blank=True, related_name="scoring_programs")
    ingestion_program = models.ForeignKey('datasets.Data', on_delete=models.SET_NULL, null=True, blank=True, related_name="ingestion_programs")
    public_data = models.ForeignKey('datasets.Data', on_delete=models.SET_NULL, null=True, blank=True, related_name="public_datas")
    starting_kit = models.ForeignKey('datasets.Data', on_delete=models.SET_NULL, null=True, blank=True, related_name="starting_kits")


class SubmissionDetails(models.Model):
    DETAILED_OUTPUT_NAMES = [
        "stdout",
        "stderr",
        "ingestion_stdout",
        "ingestion_stderr",
    ]
    name = models.CharField(max_length=50)
    data_file = models.FileField(upload_to=PathWrapper('submission_details'))
    submission = models.ForeignKey('Submission', on_delete=models.CASCADE, related_name='details')


class Submission(models.Model):
    NONE = "None"
    SUBMITTING = "Submitting"
    SUBMITTED = "Submitted"
    RUNNING = "Running"
    FINISHED = "Finished"
    FAILED = "Failed"

    STATUS_CHOICES = (
        (NONE, "None"),
        (SUBMITTING, "Submitting"),
        (SUBMITTED, "Submitted"),
        (RUNNING, "Running"),
        (FINISHED, "Finished"),
        (FAILED, "Failed"),
    )

    description = models.CharField(max_length=240, default="", blank=True, null=True)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, related_name='submission', on_delete=models.DO_NOTHING)
    status = models.CharField(max_length=128, choices=STATUS_CHOICES, default=NONE, null=False, blank=False)
    phase = models.ForeignKey(Phase, related_name='submissions', on_delete=models.CASCADE)
    appear_on_leaderboards = models.BooleanField(default=False)
    data = models.ForeignKey("datasets.Data", on_delete=models.CASCADE)
    result = models.FileField(upload_to=PathWrapper('submission_result'), null=True, blank=True)
    secret = models.UUIDField()

    # Experimental
    name = models.CharField(max_length=120, default="", null=True, blank=True)
    score = models.DecimalField(max_digits=20, decimal_places=10, default=None, null=True, blank=True)
    participant = models.ForeignKey('CompetitionParticipant', related_name='submissions', on_delete=models.CASCADE,
                                    null=True, blank=True)
    created_when = models.DateTimeField(auto_now_add=True)
    is_public = models.BooleanField(default=False)

    # uber experimental
    # track = models.IntegerField(default=1)

    def __str__(self):
        return f"{self.phase.competition.title} submission by {self.owner.username}"

    def save(self):
        created = not self.pk

        if created:
            self.status = Submission.SUBMITTING

        # Save first so we have a PK
        super(Submission, self).save()

        # Allow submissions forced back to "None" to be retried
        if created or self.status == Submission.NONE:
            from .tasks import run_submission
            run_submission.apply_async((self.pk,))

        # if not self.score:
        #     # Import here to stop circular imports
        #     from competitions import tasks
        #
        #     if self.phase:
        #         if self.phase.scoring_program:
        #             tasks.score_submission.delay(self.pk, self.phase.pk)
        #     tasks.score_submission_lazy.delay(self.pk)


class CompetitionParticipant(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, null=False, blank=False, related_name='participant',
                                on_delete=models.DO_NOTHING)
    competition = models.OneToOneField(Competition, related_name='participants', on_delete=models.CASCADE)


class Page(models.Model):
    competition = models.ForeignKey(Competition, related_name='pages', on_delete=models.CASCADE)
    title = models.TextField(max_length=255)
    content = models.TextField()
    index = models.PositiveIntegerField()


# class Leaderboard(models.Model):
#     pass
#
#
# class Leaderboard


"""
What if the competition creator adds/removes leaderboards?

what if the competition creator adds/removes columns?
"""

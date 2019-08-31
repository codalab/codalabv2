import os
import time

from django.urls import reverse

from factories import UserFactory
from ..utils import SeleniumTestCase


class TestCompetitions(SeleniumTestCase):
    def setUp(self):
        super().setUp()
        self.user = UserFactory(password='test')
        self.login(self.user.username, 'test')

    def test_competition_upload(self):
        self.get(reverse('competitions:upload'))
        self.find('input[ref="file_input"]').send_keys(os.path.join(self.test_files_dir, 'competition.zip'))
        # with self.implicit_wait_context(60):
        self.selenium.implicitly_wait(60)
        start = time.time()
        el = self.find('div .ui.success.message')
        assert el.is_displayed(), f'element not visible, waited for: {time.time() - start}'
        self.selenium.implicitly_wait(self.default_implicit_wait_time)
        self.circleci_screenshot(name='uploading_comp.png')
        comp = self.user.competitions.first()
        comp_url = reverse("competitions:detail", kwargs={"pk": comp.id})
        self.find(f'a[href="{comp_url}"]').click()
        self.assert_current_url(comp_url)
        task = comp.phases.first().tasks.first()
        created_items = [
            comp.bundle_dataset.data_file.name,
            comp.logo.name,
            task.scoring_program.data_file.name,
            task.reference_data.data_file.name,
        ]
        self.assert_storage_items_exist(*created_items)
        self.remove_items_from_storage(*created_items)

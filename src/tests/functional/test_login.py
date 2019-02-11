from django.urls import reverse

from factories import UserFactory
from ..utils import SeleniumTestCase


class TestLogin(SeleniumTestCase):

    def test_login(self):
        user = UserFactory(password="test")
        self.login(username=user.username, password='test')
        self.assertCurrentUrl(reverse("pages:home"))
        assert user.username in self.find("#user_dropdown .text").text
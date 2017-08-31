from requests import HTTPError

from six.moves.urllib.parse import urljoin

from social_core.backends.oauth import BaseOAuth2
from social_core.exceptions import AuthFailed


class CodalabOAuth2(BaseOAuth2):
    """Github OAuth authentication backend"""
    name = 'codalab'
    # API_URL = 'https://api.github.com/'
    AUTHORIZATION_URL = 'https://competitions-v2-staging-pr-17.herokuapp.com/oauth/authorize/'
    ACCESS_TOKEN_URL = 'https://competitions-v2-staging-pr-17.herokuapp.com/oauth/token/'
    ACCESS_TOKEN_METHOD = 'POST'

    def get_user_details(self, response):
        print("usa details", response)
        return {
            'username': "<username if any>,",
            'email': "<user email if any>,",
            'fullname': "<user full name if any>,",
            'first_name': "<user first name if any>,",
            'last_name': "<user last name if any>",
        }
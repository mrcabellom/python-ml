import unittest
from tests.fixtures.given_fixtures import GivenFixture

class BaseTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(BaseTestCase, self).__init__(*args, **kwargs)

    def setUp(self):
        self._given = GivenFixture()
        super(BaseTestCase, self).setUp()

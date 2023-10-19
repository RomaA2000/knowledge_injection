from unittest import TestCase
from reply import test_data
from ..utils import to_old_rep

class TestTweet(TestCase):
    def test_parse(self):
        json_data = test_data
        to_old_rep(json_data)

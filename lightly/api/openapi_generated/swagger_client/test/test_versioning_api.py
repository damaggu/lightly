"""
    Lightly API

    Lightly.ai enables you to do self-supervised learning in an easy and intuitive way. The lightly.ai OpenAPI spec defines how one can interact with our REST API to unleash the full potential of lightly.ai  # noqa: E501

    The version of the OpenAPI document: 1.0.0
    Contact: support@lightly.ai
    Generated by: https://openapi-generator.tech
"""


import unittest

import lightly.api.openapi_generated.swagger_client
from lightly.api.openapi_generated.swagger_client.api.versioning_api import VersioningApi  # noqa: E501


class TestVersioningApi(unittest.TestCase):
    """VersioningApi unit test stubs"""

    def setUp(self):
        self.api = VersioningApi()  # noqa: E501

    def tearDown(self):
        pass

    def test_get_latest_pip_version(self):
        """Test case for get_latest_pip_version

        """
        pass

    def test_get_minimum_compatible_pip_version(self):
        """Test case for get_minimum_compatible_pip_version

        """
        pass


if __name__ == '__main__':
    unittest.main()

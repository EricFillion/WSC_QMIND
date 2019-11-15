# disable pylint TODO warning
# pylint: disable=W0511
""" Testing module for HappyXLNET"""
import pytest

from happy_transformer.happy_xlnet import HappyXLNET
from testing.standard_test_data import test_data


@pytest.mark.xfail
@pytest.mark.parametrize("test_string, options, expected_token, "
                         "expected_score_threshold",
                         [test_data[0]['data'], test_data[1]['data']],
                         ids=[test_data[0]['id'], test_data[1]['id']])
def test_predict_mask(test_string, options, expected_token,
                      expected_score_threshold):
    """
    Tests the method predict_mask in HappyXLNET()

    """
    # TODO make the return token from test_predict_mask a string
    # TODO create a test for probs
    model = HappyXLNET()
    token_dct = model.predict_mask(test_string, options)
    assert token_dct[0]['word'] == expected_token
    assert token_dct[0]['score'] > expected_score_threshold
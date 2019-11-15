# disable pylint TODO warning
# pylint: disable=W0511
""" Testing module for HappyXLM"""
import pytest

from happy_transformer.happy_xlm import HappyXLM
from testing.standard_test_data import test_data


@pytest.mark.xfail
@pytest.mark.parametrize("test_string, options, expected_token, "
                         "expected_score_threshold",
                         [test_data[0]['data'], test_data[1]['data']],
                         ids=[test_data[0]['id'], test_data[1]['id']])
def test_predict_mask(test_string, options, expected_token,
                      expected_score_threshold):
    """
    Tests the method predict_mask in HappyXLM()

    """
    # TODO make the return token from test_predict_mask a string
    # TODO create a test for probs
    expected_score_threshold = 2.38093e-05  # TODO: overload for now
    model = HappyXLM()
    token_dct = model.predict_mask(test_string, options)
    assert token_dct[0]['word'] == expected_token
    assert token_dct[0]['score'] > expected_score_threshold
OK_FORMAT = True

test = {   'name': 'q1',
    'points': 5,
    'suites': [   {   'cases': [   {   'code': ">>> assert isinstance(T_series, np.ndarray), 'The matrix should be a numpy array.'\n"
                                               ">>> assert isinstance(error_series, (float, np.floating)), 'The norm should be a float.'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> expected_shape = (3, 3)\n>>> assert T_series.shape == expected_shape, f'Expected shape {expected_shape}, but got {T_series.shape}.'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': ">>> assert np.allclose(T_series, T_exact), 'Power series approximation does not match analytical values. If your number of iterations (n) is low, try "
                                               "raising it.'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> import math\n'
                                               ">>> expected_norm = np.linalg.norm(T_series - T_exact, ord='fro')\n"
                                               ">>> assert math.isclose(error_series, expected_norm), 'The Frobenius norm calculation is incorrect.'\n",
                                       'hidden': False,
                                       'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}

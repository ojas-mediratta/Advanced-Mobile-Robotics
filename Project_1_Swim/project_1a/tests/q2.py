OK_FORMAT = True

test = {   'name': 'q2',
    'points': 5,
    'suites': [   {   'cases': [   {   'code': ">>> assert isinstance(H1, np.ndarray), 'H1 should be a numpy array.'\n"
                                               ">>> assert isinstance(H2, np.ndarray), 'H2 should be a numpy array.'\n"
                                               ">>> assert isinstance(adj_T2inv, np.ndarray), 'Analytic Adjoint should be a numpy array.'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {   'code': '>>> expected_shape = (3, 3)\n'
                                               ">>> assert H1.shape == expected_shape, f'Expected H1 shape {expected_shape}, but got {H1.shape}.'\n"
                                               ">>> assert H2.shape == expected_shape, f'Expected H2 shape {expected_shape}, but got {H2.shape}.'\n"
                                               ">>> assert adj_T2inv.shape == expected_shape, f'Expected Analytic Adjoint shape {expected_shape}, but got {adj_T2inv.shape}.'\n",
                                       'hidden': False,
                                       'locked': False},
                                   {'code': ">>> assert np.allclose(H1, adj_T2inv), 'H1 and adj_T2inv values do not match.'\n", 'hidden': False, 'locked': False},
                                   {'code': ">>> assert np.allclose(H2, np.eye(3)), 'H2 is not correct.'\n", 'hidden': False, 'locked': False}],
                      'scored': True,
                      'setup': '',
                      'teardown': '',
                      'type': 'doctest'}]}

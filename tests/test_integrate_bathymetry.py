import numpy as np

from bathytools.bathymetry_sources import refine_array


def test_refine_array1():
    a = np.linspace(0, 10, 11)
    b = np.linspace(0, 5, 6) + 1.2

    v, v_indices = refine_array(b, a)

    expected_v = np.empty((len(a) + len(b),), dtype=b.dtype)
    expected_v[0] = 0.0
    expected_v[1 : 2 * len(b) + 1 : 2] = a[1 : len(b) + 1]
    expected_v[2 : 2 * len(b) + 2 : 2] = b
    expected_v[2 * len(b) + 1 :] = a[len(b) + 1 :]

    assert np.all(v == expected_v)


def test_refine_array2():
    a = np.linspace(0, 10, 11)
    b = np.linspace(0, 0.9, 10) + 0.05

    v, v_indices = refine_array(b, a)

    expected_v = np.empty((len(a) + len(b),), dtype=a.dtype)
    expected_v[0] = a[0]
    expected_v[1 : len(b) + 1] = b
    expected_v[len(b) + 1 :] = a[1:]

    expected_v_indices = np.empty((len(a) + len(b),), dtype=np.int32)
    expected_v_indices[: len(b) + 1] = 0
    expected_v_indices[len(b) + 1 :] = np.arange(1, len(a))

    assert np.all(v == expected_v)
    assert np.all(v_indices == expected_v_indices)

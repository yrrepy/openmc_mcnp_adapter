from textwrap import dedent

import numpy as np
from pytest import mark, param
from openmc_mcnp_adapter import mcnp_str_to_model


# Cell 2 is the lattice cell (universe 50, material 8). Universes 2 through 8
# are filled with materials 1 through 7, universe 9 is void.
_TEMPLATE = dedent("""\
    lattice
    1    0          -900   fill=50
    2    8 -1.0     {region}   lat={lat} u=50
    {fill}
    10   1 -1.0     -901   u=2
    11   2 -1.0     -901   u=3
    12   3 -1.0     -901   u=4
    13   4 -1.0     -901   u=5
    14   5 -1.0     -901   u=6
    15   6 -1.0     -901   u=7
    16   7 -1.0     -901   u=8
    17   0          -901   u=9
    20   0           900

    {surfaces}
    900  so 100.0
    901  so 1000.0

    m1   1001.80c   1.0
    m2   8016.80c   1.0
    m3   26056.80c  1.0
    m4   2004.80c   1.0
    m5   3007.80c   1.0
    m6   5011.80c   1.0
    m7   13027.80c  1.0
    m8   92235.80c  1.0
    """)

# Material found in each fill universe; universe 50 is the lattice cell itself
_MATERIAL = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: None, 50: 8}

# Asymmetric 5x5 array with the first index varying fastest
_FILL_2D = [9, 9, 2, 3, 9,
            9, 4, 5, 6, 9,
            7, 8, 50, 2, 3,
            9, 4, 6, 9, 9,
            9, 5, 7, 9, 9]

# Three 3x3 layers, each with its own universes, with the first index varying
# fastest
_FILL_3D = [2, 3, 2, 2, 2, 2, 2, 2, 2,
            4, 4, 4, 5, 50, 4, 4, 4, 4,
            6, 6, 6, 6, 6, 6, 7, 6, 6]

_RECT_PLANES = """\
201  px  1.0
202  px -1.0
203  py  1.5
204  py -1.5
205  pz  2.0
206  pz -2.0"""


def _fill_card(indices, univ_ids, per_line=9):
    """Build the FILL card of a lattice cell as continuation lines."""
    lines = ['     fill=' + indices]
    for i in range(0, len(univ_ids), per_line):
        ids = ' '.join(str(u) for u in univ_ids[i:i + per_line])
        lines.append('     ' + ids)
    return '\n'.join(lines)


def _convert(lat, region, surfaces, indices, univ_ids):
    """Convert the template with the given lattice cell and FILL array."""
    mcnp_str = _TEMPLATE.format(
        lat=lat, region=region, surfaces=surfaces,
        fill=_fill_card(indices, univ_ids))
    return mcnp_str_to_model(mcnp_str)


def _assert_elements(model, ranges, center, vectors, univ_ids):
    """Check the material found at the center of every MCNP lattice element."""
    (i1, i2), (j1, j2), (k1, k2) = ranges
    a1, a2, a3 = (np.asarray(v, dtype=float) for v in vectors)
    n = 0
    for k in range(k1, k2 + 1):
        for j in range(j1, j2 + 1):
            for i in range(i1, i2 + 1):
                point = np.asarray(center, dtype=float) + i*a1 + j*a2 + k*a3
                cell = model.geometry.find(point)[-1]
                found = None if cell.fill is None else cell.fill.id
                assert found == _MATERIAL[univ_ids[n]], \
                    'element [{},{},{}] at {}'.format(i, j, k, point)
                n += 1


@mark.parametrize("surfaces,region,center,vectors", [
    param(_RECT_PLANES, '-201 202 -203 204', (0., 0., 0.),
          ((2., 0., 0.), (0., 3., 0.), (0., 0., 0.)), id='planes'),
    # The first pair of surfaces gives the direction of the first index
    param(_RECT_PLANES, '-203 204 -201 202', (0., 0., 0.),
          ((0., 3., 0.), (2., 0., 0.), (0., 0., 0.)), id='y-planes-first'),
])
def test_rect_lattice(surfaces, region, center, vectors):
    ranges = ((-2, 2), (-2, 2), (0, 0))
    model = _convert(1, region, surfaces, '-2:2 -2:2 0:0', _FILL_2D)
    _assert_elements(model, ranges, center, vectors, _FILL_2D)


def test_rect_lattice_3d():
    # The z planes listed first, bottom plane first, put the first index
    # along -z
    ranges = ((-1, 1), (-1, 1), (-1, 1))
    vectors = ((0., 0., -4.), (2., 0., 0.), (0., 3., 0.))
    model = _convert(1, '206 -205 -201 202 -203 204', _RECT_PLANES,
                     '-1:1 -1:1 -1:1', _FILL_3D)
    _assert_elements(model, ranges, (0., 0., 0.), vectors, _FILL_3D)

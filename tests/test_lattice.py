from textwrap import dedent

import numpy as np
from pytest import mark, param
from openmc_mcnp_adapter import mcnp_str_to_model


SQRT3 = 1.7320508076

# Cell 2 is the hexagonal lattice cell (universe 50, material 8). Universes 2
# through 8 are filled with materials 1 through 7, universe 9 is void.
_TEMPLATE = dedent("""\
    hexagonal lattice
    1    0          -900   fill=50
    2    8 -1.0     {region}   lat=2 u=50
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

# Manual Listing 10.8: hexagon with an apothem of 1 cm centered at the origin
_PLANES = """\
301  px  1.0
302  px -1.0
303  p   1.0  1.7320508076  0.0  2.0
304  p  -1.0  1.7320508076  0.0  2.0
305  p   1.0  1.7320508076  0.0 -2.0
306  p  -1.0  1.7320508076  0.0 -2.0"""

# Same hexagon with R along y
_PLANES_Y = """\
311  py  1.0
312  py -1.0
313  p  -1.7320508076  1.0  0.0  2.0
314  p  -1.7320508076  1.0  0.0 -2.0
315  p  -1.7320508076 -1.0  0.0  2.0
316  p  -1.7320508076 -1.0  0.0 -2.0"""

def _fill_card(indices, univ_ids, per_line=5):
    """Build the FILL card of a lattice cell as continuation lines."""
    lines = ['     fill=' + indices]
    for i in range(0, len(univ_ids), per_line):
        lines.append('     ' + ' '.join(str(u) for u in univ_ids[i:i + per_line]))
    return '\n'.join(lines)


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


@mark.parametrize("surfaces,region,center,a1,a2", [
    param(_PLANES, '-301 302 -303 305 -304 306', (0., 0., 0.),
          (2., 0., 0.), (1., SQRT3, 0.), id='planes'),
    param('1  rhp  2 0.5 -5   0 0 10   1 0 0', '-1', (2., 0.5, 0.),
          (2., 0., 0.), (1., SQRT3, 0.), id='rhp-off-center'),
    param(_PLANES_Y, '-311 312 -313 314 -315 316', (0., 0., 0.),
          (0., 2., 0.), (-SQRT3, 1., 0.), id='r-along-y'),
])
def test_hex_lattice(surfaces, region, center, a1, a2):
    ranges = ((-2, 2), (-2, 2), (0, 0))
    mcnp_str = _TEMPLATE.format(
        region=region, surfaces=surfaces,
        fill=_fill_card('-2:2 -2:2 0:0', _FILL_2D))
    model = mcnp_str_to_model(mcnp_str)

    _assert_elements(model, ranges, center, (a1, a2, (0., 0., 0.)), _FILL_2D)


def test_hex_lattice_3d():
    # Three axial layers, each with its own pair of universes, with the height
    # vector of the prism pointing along -z
    univ_ids = [2, 2, 2, 2, 3, 2, 2, 2, 2,
                4, 4, 4, 4, 5, 4, 4, 4, 4,
                6, 6, 6, 6, 7, 6, 6, 6, 6]
    ranges = ((-1, 1), (-1, 1), (-1, 1))
    vectors = ((2., 0., 0.), (1., -SQRT3, 0.), (0., 0., -10.))
    mcnp_str = _TEMPLATE.format(
        region='-1', surfaces='1  rhp  0 0 5   0 0 -10   1 0 0',
        fill=_fill_card('-1:1 -1:1 -1:1', univ_ids, per_line=9))
    model = mcnp_str_to_model(mcnp_str)

    _assert_elements(model, ranges, (0., 0., 0.), vectors, univ_ids)

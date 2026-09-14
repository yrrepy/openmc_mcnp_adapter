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

SQRT3 = 1.7320508076

# Manual Listing 10.8: hexagon with an apothem of 1 cm centered at the origin
_HEX_PLANES = """\
301  px  1.0
302  px -1.0
303  p   1.0  1.7320508076  0.0  2.0
304  p  -1.0  1.7320508076  0.0  2.0
305  p   1.0  1.7320508076  0.0 -2.0
306  p  -1.0  1.7320508076  0.0 -2.0"""

# Same hexagon with R along y
_HEX_PLANES_Y = """\
311  py  1.0
312  py -1.0
313  p  -1.7320508076  1.0  0.0  2.0
314  p  -1.7320508076  1.0  0.0 -2.0
315  p  -1.7320508076 -1.0  0.0  2.0
316  p  -1.7320508076 -1.0  0.0 -2.0"""


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
    param(_RECT_PLANES, '202 -201 204 -203', (0., 0., 0.),
          ((-2., 0., 0.), (0., -3., 0.), (0., 0., 0.)), id='reversed'),
    param('1  rpp  0 2  0.5 3.5  -2 2', '-1', (1., 2., 0.),
          ((2., 0., 0.), (0., 3., 0.), (0., 0., 4.)), id='rpp'),
])
def test_rect_lattice(surfaces, region, center, vectors):
    ranges = ((-2, 2), (-2, 2), (0, 0))
    model = _convert(1, region, surfaces, '-2:2 -2:2 0:0', _FILL_2D)
    _assert_elements(model, ranges, center, vectors, _FILL_2D)


@mark.parametrize("region,vectors", [
    param('-201 202 -203 204 -205 206',
          ((2., 0., 0.), (0., 3., 0.), (0., 0., 4.)), id='z-up'),
    param('-201 202 -203 204 206 -205',
          ((2., 0., 0.), (0., 3., 0.), (0., 0., -4.)), id='z-down'),
    param('-205 206 -201 202 -203 204',
          ((0., 0., 4.), (2., 0., 0.), (0., 3., 0.)), id='z-planes-first'),
])
def test_rect_lattice_3d(region, vectors):
    ranges = ((-1, 1), (-1, 1), (-1, 1))
    model = _convert(1, region, _RECT_PLANES, '-1:1 -1:1 -1:1', _FILL_3D)
    _assert_elements(model, ranges, (0., 0., 0.), vectors, _FILL_3D)


def test_rect_lattice_infinite():
    model = _convert(1, '-201 202 -203 204', _RECT_PLANES, '', [3])
    lattice = model.geometry.get_all_lattices()[50]
    assert lattice.outer is not None
    assert model.geometry.find((45.3, -20.1, 0.))[-1].fill.id == 2


@mark.parametrize("surfaces,region,center,vectors", [
    param(_HEX_PLANES, '-301 302 -303 305 -304 306', (0., 0., 0.),
          ((2., 0., 0.), (1., SQRT3, 0.), (0., 0., 0.)), id='planes'),
    param('1  rhp  2 0.5 -5   0 0 10   1 0 0', '-1', (2., 0.5, 0.),
          ((2., 0., 0.), (1., SQRT3, 0.), (0., 0., 10.)), id='rhp-off-center'),
    param(_HEX_PLANES_Y, '-311 312 -313 314 -315 316', (0., 0., 0.),
          ((0., 2., 0.), (-SQRT3, 1., 0.), (0., 0., 0.)), id='r-along-y'),
])
def test_hex_lattice(surfaces, region, center, vectors):
    ranges = ((-2, 2), (-2, 2), (0, 0))
    model = _convert(2, region, surfaces, '-2:2 -2:2 0:0', _FILL_2D)
    _assert_elements(model, ranges, center, vectors, _FILL_2D)


def test_hex_lattice_3d():
    # Height vector of the prism pointing along -z
    ranges = ((-1, 1), (-1, 1), (-1, 1))
    vectors = ((2., 0., 0.), (1., -SQRT3, 0.), (0., 0., -10.))
    model = _convert(2, '-1', '1  rhp  0 0 5   0 0 -10   1 0 0',
                     '-1:1 -1:1 -1:1', _FILL_3D)
    _assert_elements(model, ranges, (0., 0., 0.), vectors, _FILL_3D)


def test_hex_lattice_infinite():
    model = _convert(2, '-301 302 -303 305 -304 306', _HEX_PLANES, '', [3])
    lattice = model.geometry.get_all_lattices()[50]
    assert lattice.outer is not None
    assert model.geometry.find((45.3, -20.1, 0.))[-1].fill.id == 2


@mark.parametrize("inner_first", [True, False])
def test_nested_lattice(inner_first):
    # 3x3 outer lattice (pitch 3) whose elements are universe 2, itself a 3x3
    # lattice (pitch 1) with universe 3 at its center and universe 4 elsewhere
    outer = "2 0 -2 LAT=1 U=1 FILL=-1:1 -1:1 0:0 2 2 2 2 2 2 2 2 2"
    inner = "3 0 -3 LAT=1 U=2 FILL=-1:1 -1:1 0:0 4 4 4 4 3 4 4 4 4"
    lattices = f"{inner}\n    {outer}" if inner_first else f"{outer}\n    {inner}"
    mcnp_str = dedent(f"""
    title
    1 0 -1 FILL=1
    {lattices}
    4 1 -1.0 -4 U=3
    5 2 -2.0 +4 U=3
    6 2 -2.0 -5 U=4

    1 rpp -4.5 4.5 -4.5 4.5 -4.5 4.5
    2 rpp -1.5 1.5 -1.5 1.5 -1.5 1.5
    3 rpp -0.5 0.5 -0.5 0.5 -1.5 1.5
    4 so 0.2
    5 so 10.0

    m1   1001.80c  1.0
    m2   1002.80c  1.0
    """)
    geometry = mcnp_str_to_model(mcnp_str).geometry

    # Center of the inner center element of the outer element at (3, 0, 0)
    assert geometry.find((3.0, 0.0, 0.0))[-1].fill.id == 1
    # One inner pitch over is the surrounding universe 4
    assert geometry.find((4.0, 0.0, 0.0))[-1].fill.id == 2
    # Same for the outer element at (0, 3, 0)
    assert geometry.find((0.0, 3.0, 0.0))[-1].fill.id == 1

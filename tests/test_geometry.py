from textwrap import dedent

from pytest import mark, approx, param
from openmc_mcnp_adapter import mcnp_str_to_model


@mark.parametrize("whitespace", ["", " ", "\t"])
def test_cell_complement(whitespace):
    # Cell 2 corresponds to r < 2 intersected with z > 0
    mcnp_str = dedent(f"""
    title
    100  1 1.0  +1 : -2
    2    1 1.0  #{whitespace}100

    1  so 2.0
    2  pz 0.0

    m1   1001.80c  1.0
    """)
    model = mcnp_str_to_model(mcnp_str)
    cell = model.geometry.get_all_cells()[2]

    # Check various points
    assert (0., 0., 0.1) in cell.region
    assert (0., 0., -0.1) not in cell.region
    assert (0., 0., 1.99) in cell.region
    assert (0., 0., 2.01) not in cell.region
    assert (1., 1., 1.) in cell.region
    assert (2., 0., 1.) not in cell.region


def test_likenbut():
    mcnp_str = dedent("""
    title
    1   1 -1.0  -1
    2   LIKE 1 BUT MAT=2 RHO=-2.0 TRCL=(2.0 0.0 0.0)

    1   so 1.0

    m1   1001.80c  1.0
    m2   1002.80c  1.0
    """)
    model = mcnp_str_to_model(mcnp_str)
    cell = model.geometry.get_all_cells()[2]

    # Material should be changed to m2
    mat = cell.fill
    assert 'H2' in mat.get_nuclide_densities()

    # Density should be 2.0 g/cm3
    assert mat.get_mass_density() == approx(2.0)

    # Points should correspond to sphere of r=1 centered at (2, 0, 0)
    assert (2.0, 0.0, 0.0) in cell.region
    assert (0.0, 0.0, 0.0) not in cell.region
    assert (2.0, 0.9, 0.0) in cell.region
    assert (2.0, 1.1, 0.0) not in cell.region


@mark.parametrize(
    "cell_card, surface_cards, points_inside, points_outside",
    [
        (
            "1   1 -1.0  -1 TRCL=(2.0 0.0 0.0)",
            ("1   so 1.0",),
            [
                (2.0, 0.0, 0.0),
                (2.0, 0.9, 0.0),
                (2.0, 0.0, 0.9),
            ],
            [
                (0.9, 0.0, 0.0),
                (2.0, 1.1, 0.0),
                (2.0, 0.0, 1.1),
            ],
        ),
        (
            "1 0 -1 TRCL=(1.0 0.0 0.0 0.0 -1.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0)",
            ("1 rpp -0.5 0.5 -0.25 0.25 -0.1 0.1",),
            [
                (1.0, 0.0, 0.0),
                (1.2, 0.0, 0.0),
                (1.0, 0.4, 0.0),
                (1.0, -0.4, 0.0),
            ],
            [
                (0.7, 0.0, 0.0),
                (1.3, 0.0, 0.0),
                (1.0, 0.6, 0.0),
                (1.0, 0.0, 0.2),
            ],
        ),
        (
            "1 0 -1 *TRCL=(1.0 0.0 0.0 90.0 180.0 90.0 0.0 90.0 90.0 90.0 90.0 0.0)",
            ("1 rpp -0.5 0.5 -0.25 0.25 -0.1 0.1",),
            [
                (1.0, 0.0, 0.0),
                (1.2, 0.0, 0.0),
                (1.0, 0.4, 0.0),
                (1.0, -0.4, 0.0),
            ],
            [
                (0.7, 0.0, 0.0),
                (1.3, 0.0, 0.0),
                (1.0, 0.6, 0.0),
                (1.0, 0.0, 0.2),
            ],
        ),
        (
            "1 0 -1 TRCL=(0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 -1)",
            ("1 rpp -0.5 0.5 -0.25 0.25 -0.1 0.1",),
            [
                (-0.4, 0.1, 0.0),
                (0.0, 0.0, 0.0),
                (0.49, -0.24, -0.05),
            ],
            [
                (-0.6, 0.0, 0.0),
                (0.0, 0.5, 0.0),
                (0.0, 0.0, 0.2),
            ],
        ),
    ],
)
def test_trcl(cell_card, surface_cards, points_inside, points_outside):
    surface_block = "\n".join(surface_cards)
    mcnp_str = dedent(f"""
    title
    {cell_card}

    {surface_block}

    m1   1001.80c  1.0
    """)
    model = mcnp_str_to_model(mcnp_str)
    cell = model.geometry.get_all_cells()[1]

    for point in points_inside:
        assert point in cell.region

    for point in points_outside:
        assert point not in cell.region


def test_trcl_macrobody():
    mcnp_str = dedent("""
    title
    1 0 -1 trcl=(2.0 0.0 0.0)

    1 rpp -1.0 1.0 -1.0 1.0 -1.0 1.0

    m1     1001.80c  1.0
    """)
    model = mcnp_str_to_model(mcnp_str)
    cell = model.geometry.get_all_cells()[1]
    assert (1.5, 0., 0.) in cell.region
    assert (0., 0., 0.) not in cell.region


@mark.parametrize(
    "keywords",
    [
        "FILL=10 TRCL=(2.0 0.0 0.0)",
        "FILL=10(2.0 0.0 0.0)",
        "FILL=10 TRCL=(2.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0)",
        "FILL=10(2.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0)",
        "FILL=10 *TRCL=(2.0 0.0 0.0 0.0 90.0 90.0 90.0 0.0 90.0 90.0 90.0 0.0)",
        "*FILL=10(2.0 0.0 0.0 0.0 90.0 90.0 90.0 0.0 90.0 90.0 90.0 0.0)",
        "FILL=10 TRCL=(2.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 1)",
        "FILL=10 TRCL=(-2.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 -1)",
        "FILL=10(2.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 1)",
        "FILL=10(-2.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 -1)",
        "FILL=10 *TRCL=(2.0 0.0 0.0 0.0 90.0 90.0 90.0 0.0 90.0 90.0 90.0 0.0 1)",
        "FILL=10 *TRCL=(-2.0 0.0 0.0 0.0 90.0 90.0 90.0 0.0 90.0 90.0 90.0 0.0 -1)",
        "*FILL=10(2.0 0.0 0.0 0.0 90.0 90.0 90.0 0.0 90.0 90.0 90.0 0.0 1)",
        "*FILL=10(-2.0 0.0 0.0 0.0 90.0 90.0 90.0 0.0 90.0 90.0 90.0 0.0 -1)",
        "FILL=10(1)",
        "FILL=10(2)",
        "FILL=10(3)",
    ]
)
def test_fill_transformation(keywords):
    mcnp_str = dedent(f"""
    title
    1 0 -1 {keywords}
    2 0 -2 U=10
    3 0 +2 U=10

    1 so 10.0
    2 so 1.0

    m1     1001.80c  1.0
    tr1    2.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0 1
    *tr2   -2.0 0.0 0.0 0.0 90.0 90.0 90.0 0.0 90.0 90.0 90.0 0.0 -1
    tr3    2.0 0.0 0.0
    """)
    model = mcnp_str_to_model(mcnp_str)
    geometry = model.geometry
    cells = geometry.get_all_cells()

    # Make sure that the cells in universe 10 were shifted
    assert geometry.find((2.0, 0.0, 0.0))[-1] is cells[2]
    assert geometry.find((4.0, 0.0, 0.0))[-1] is cells[3]
    assert geometry.find((0.0, 0.0, 0.0))[-1] is cells[3]


def test_cell_volume():
    mcnp_str = dedent("""
    title
    1 0 -1  VOL=5.0

    1 so 1.0

    m1   1001.80c  1.0
    """)
    model = mcnp_str_to_model(mcnp_str)
    cell = model.geometry.get_all_cells()[1]
    assert cell.volume == 5.0


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


@mark.parametrize("k,bottom_first", [(0, False), (2, False), (-3, False), (2, True)],
                  ids=['k0', 'k2', 'k-3', 'k2-bottom-first'])
def test_single_layer_lattice(k, bottom_first):
    # The single layer at axial index k of a lattice whose element spans
    # z = 0 to 1 becomes a 2D lattice whose universes are translated to the
    # layer. Listing the bottom plane first puts index k below the element.
    z_planes = "16 -15" if bottom_first else "-15 16"
    z0 = -k if bottom_first else k
    mcnp_str = dedent(f"""
    title
    1 0 -1 FILL=2
    2 0 -11 12 -13 14 {z_planes} LAT=1 U=2 FILL=-1:1 -1:1 {k}:{k} 3 3 3 3 3 3 3 3 3
    3 1 -1.0 -21 U=3
    4 2 -2.0 +21 U=3
    9 0 1

    1 rpp -1.5 1.5 -1.5 1.5 {z0} {z0 + 1}
    11 px 0.5
    12 px -0.5
    13 py 0.5
    14 py -0.5
    15 pz 1.0
    16 pz 0.0
    21 s 0.2 0.1 0.6 0.3

    m1   1001.80c  1.0
    m2   1002.80c  1.0
    """)
    geometry = mcnp_str_to_model(mcnp_str).geometry
    assert geometry.get_all_lattices()[2].ndim == 2

    # Center of the sphere in two elements, and a point below the sphere
    assert geometry.find((1.2, 0.1, z0 + 0.6))[-1].fill.id == 1
    assert geometry.find((0.2, -0.9, z0 + 0.6))[-1].fill.id == 1
    assert geometry.find((0.2, 0.1, z0 + 0.1))[-1].fill.id == 2

# -*- coding: utf-8 -*-
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.

"""Tests for GaussianSplatVisual."""

import numpy as np
import pytest

from vispy import scene, use, visuals
from vispy.testing import (TestingCanvas, has_pyopengl, requires_application,
                           requires_pyopengl, run_tests_if_main)


# the data-only tests don't need gl+, so only switch when PyOpenGL is around;
# the drawing tests are marked requires_pyopengl
def setup_module(module):
    if has_pyopengl():
        use(gl='gl+')


def teardown_module(module):
    if has_pyopengl():
        use(gl='gl2')


def _make_splats(n=50, seed=0):
    rng = np.random.RandomState(seed)
    positions = rng.randn(n, 3).astype(np.float32)
    a = (rng.randn(n, 3, 3) * 0.1).astype(np.float32)
    # symmetric positive-definite covariances
    covariances = np.einsum('nij,nkj->nik', a, a) + np.eye(3, dtype=np.float32) * 0.02
    colors = rng.rand(n, 4).astype(np.float32)  # RGBA
    return positions, covariances, colors


def test_splat_data_roundtrip():
    positions, covariances, colors = _make_splats()
    v = visuals.GaussianSplatVisual(positions, covariances, colors)

    np.testing.assert_array_equal(v.positions, positions)
    # a plain RGBA color is stored as the degree-0 SH case, so it round-trips
    # through (rgb - 0.5) / C0 up to float32 precision
    assert v.sh_degree == 0
    np.testing.assert_allclose(v.colors, colors, atol=1e-6)
    # covariances are packed to the triangle and reassembled symmetrically
    np.testing.assert_allclose(v.covariances, covariances, atol=1e-6)
    assert np.allclose(v.covariances, np.transpose(v.covariances, (0, 2, 1)))


def test_splat_compute_bounds():
    positions, covariances, colors = _make_splats()
    v = visuals.GaussianSplatVisual(positions, covariances, colors)
    for axis in range(3):
        lo, hi = v._compute_bounds(axis, None)
        assert lo == positions[:, axis].min()
        assert hi == positions[:, axis].max()


def test_splat_set_data_partial():
    positions, covariances, colors = _make_splats()
    v = visuals.GaussianSplatVisual(positions, covariances, colors)
    v.set_data(colors=np.clip(colors * 0.5, 0, 1))
    np.testing.assert_allclose(v.colors, np.clip(colors * 0.5, 0, 1), atol=1e-6)
    # untouched arrays are preserved
    np.testing.assert_array_equal(v.positions, positions)


def test_splat_eps():
    positions, covariances, colors = _make_splats()
    # the finite-difference step is ~1% of the bounding-box extent
    v = visuals.GaussianSplatVisual(positions, covariances, colors)
    extent = float((positions.max(0) - positions.min(0)).max())
    assert np.isclose(v.shared_program['u_eps'], 1e-2 * extent)
    # scaling the scene scales the step proportionally (numerically stable)
    v2 = visuals.GaussianSplatVisual(positions * 1000, covariances, colors)
    assert np.isclose(v2.shared_program['u_eps'], 1e-2 * extent * 1000)


def _make_sh(n=20, degree=3, seed=1):
    k = (degree + 1) ** 2
    rng = np.random.RandomState(seed)
    return (rng.randn(n, k, 3).astype(np.float32),
            rng.rand(n).astype(np.float32))


@pytest.mark.parametrize('degree', [0, 1, 2, 3])
def test_splat_sh_roundtrip(degree):
    n = 20
    positions, covariances, _ = _make_splats(n)
    sh, opacity = _make_sh(n, degree)
    v = visuals.GaussianSplatVisual(positions, covariances, sh,
                                    opacities=opacity)
    assert v.sh_degree == degree
    assert v.max_sh_degree == degree
    np.testing.assert_array_equal(v.sh_coeffs, sh)
    np.testing.assert_array_equal(v.opacities, opacity)
    # above degree 0, `colors` is the SH array; at degree 0 it is RGBA
    if degree == 0:
        assert v.colors.shape == (n, 4)
    else:
        np.testing.assert_array_equal(v.colors, sh)


def test_splat_rgb_plus_opacity():
    """(N, 3) RGB works as long as opacity is supplied separately."""
    positions, covariances, colors = _make_splats(10)
    v = visuals.GaussianSplatVisual(positions, covariances, colors[:, :3],
                                    opacities=colors[:, 3])
    assert v.sh_degree == 0
    np.testing.assert_allclose(v.colors, colors, atol=1e-6)

    # a scalar opacity broadcasts over all splats
    v = visuals.GaussianSplatVisual(positions, covariances, colors[:, :3],
                                    opacities=0.25)
    np.testing.assert_array_equal(v.opacities, np.full(10, 0.25, np.float32))


def test_splat_single_color_broadcast():
    """A single color applies to every splat (useful for debugging)."""
    positions, covariances, _ = _make_splats(10)

    v = visuals.GaussianSplatVisual(positions, covariances, 'white',
                                    opacities=0.3)
    assert v.sh_degree == 0
    np.testing.assert_allclose(v.colors, np.tile([1, 1, 1, 0.3], (10, 1)),
                               atol=1e-6)

    # a color may carry its own alpha, which opacity then overrides
    v = visuals.GaussianSplatVisual(positions, covariances, '#ff000080')
    np.testing.assert_allclose(v.colors[:, :3], np.tile([1, 0, 0], (10, 1)),
                               atol=1e-6)
    np.testing.assert_allclose(v.colors[:, 3], 0.5019608, atol=1e-6)

    v = visuals.GaussianSplatVisual(positions, covariances, (0, 0, 1, 0.25),
                                    opacities=0.9)
    np.testing.assert_allclose(v.colors[:, 3], 0.9, atol=1e-6)

    # and it can replace per-splat colors after the fact
    v.set_data(colors=(0, 1, 0))
    np.testing.assert_allclose(v.colors[:, :3], np.tile([0, 1, 0], (10, 1)),
                               atol=1e-6)


def test_splat_single_color_invalid():
    positions, covariances, _ = _make_splats(10)
    with pytest.raises(ValueError):
        visuals.GaussianSplatVisual(positions, covariances, (1, 0),
                                    opacities=1.0)
    with pytest.raises(ValueError):
        visuals.GaussianSplatVisual(positions, covariances, 'notacolor',
                                    opacities=1.0)


def test_splat_colors_required():
    positions, covariances, _ = _make_splats(10)
    with pytest.raises(TypeError):
        visuals.GaussianSplatVisual(positions, covariances)


def test_splat_opacity_conflicts_with_rgba():
    positions, covariances, colors = _make_splats(10)
    with pytest.raises(ValueError, match="alpha channel"):
        visuals.GaussianSplatVisual(positions, covariances, colors,
                                    opacities=np.ones(10, np.float32))


def test_splat_opacity_required():
    positions, covariances, _ = _make_splats(10)
    sh, _ = _make_sh(10, degree=1)
    with pytest.raises(ValueError, match="opacities"):
        visuals.GaussianSplatVisual(positions, covariances, sh)


def test_splat_sh_degree_default_tracks_data():
    """With no explicit degree, the visual renders every coefficient given."""
    positions, covariances, _ = _make_splats(20)
    sh, opacity = _make_sh(20, degree=3)
    v = visuals.GaussianSplatVisual(positions, covariances, sh,
                                    opacities=opacity)
    assert v.sh_degree == 3

    # new colors at a different degree are rendered in full
    sh1, _ = _make_sh(20, degree=1, seed=2)
    v.set_data(colors=sh1)
    assert v.sh_degree == v.max_sh_degree == 1


def test_splat_sh_degree_toggle():
    positions, covariances, _ = _make_splats(20)
    sh, opacity = _make_sh(20, degree=3)
    v = visuals.GaussianSplatVisual(positions, covariances, sh,
                                    opacities=opacity)

    v.sh_degree = 0
    assert v.sh_degree == 0
    # the coefficients are kept host side, so the change is reversible ...
    np.testing.assert_array_equal(v.sh_coeffs, sh)
    v.sh_degree = 3
    assert v.sh_degree == 3


def test_splat_sh_degree_shrinks_color_texture():
    """Lowering the degree shrinks the color texture; geometry is untouched."""
    n = 20000                    # enough that row padding stays negligible
    positions, covariances, _ = _make_splats(n)
    sh, opacity = _make_sh(n, degree=3)
    v = visuals.GaussianSplatVisual(positions, covariances, sh,
                                    opacities=opacity)

    def texels(tex):
        return tex.shape[0] * tex.shape[1]

    geom_texels = texels(v._geom_tex)
    sizes = {}
    # the color texture holds exactly n * K texels, plus at most a partial row
    for degree, k in ((3, 16), (2, 9), (1, 4), (0, 1)):
        v.sh_degree = degree
        sizes[degree] = texels(v._sh_tex)
        assert n * k <= sizes[degree] < n * k + v._sh_tex.shape[1]
        # geometry always holds exactly n records, whatever the row layout
        assert n * 3 <= texels(v._geom_tex) < n * 3 + v._geom_tex.shape[1]

    # ... so dropping the degree really does shrink it, monotonically
    assert sizes[3] > sizes[2] > sizes[1] > sizes[0]
    # and the geometry texture only ever holds the same n records
    assert geom_texels >= n * 3


def test_splat_reupload_after_inplace_mutation():
    """Re-passing the same (mutated) array still re-uploads it."""
    n = 100
    positions, covariances, colors = _make_splats(n)
    v = visuals.GaussianSplatVisual(positions, covariances, colors)

    uploads = []
    v._geom_tex.set_data = lambda a, **kw: uploads.append(a)

    positions *= 2.0                     # mutated in place, same object
    v.set_data(positions=positions)
    assert len(uploads) == 1
    np.testing.assert_array_equal(uploads[0][0, 0, :3], positions[0])


def test_splat_texture_packing():
    """Each splat's record lands in its own span of its texture row."""
    n = 2000
    positions, covariances, _ = _make_splats(n)
    sh, opacity = _make_sh(n, degree=1)
    v = visuals.GaussianSplatVisual(positions, covariances, sh,
                                    opacities=opacity)

    uploads = {}
    v._geom_tex.set_data = lambda a, **kw: uploads.__setitem__('geom', a)
    v._sh_tex.set_data = lambda a, **kw: uploads.__setitem__('sh', a)
    v.set_data(positions=positions, colors=sh)

    geom = uploads['geom']
    per_row = v._per_row
    for i in (0, 1, n // 2, n - 1):
        row, col0 = int(i // per_row), int((i % per_row) * 3)
        np.testing.assert_array_equal(geom[row, col0, :3], positions[i])
        assert geom[row, col0, 3] == opacity[i]
        # covariance is packed as (S00, S01, S02, S11), (S12, S22, _, _)
        np.testing.assert_array_equal(geom[row, col0 + 1, :3], covariances[i, 0])
        assert geom[row, col0 + 1, 3] == covariances[i, 1, 1]
        assert geom[row, col0 + 2, 0] == covariances[i, 1, 2]
        assert geom[row, col0 + 2, 1] == covariances[i, 2, 2]

    # both textures share one splats-per-row, so one slot addresses both
    sh_tex = uploads['sh']
    for i in (0, 1, n // 2, n - 1):
        row, col0 = int(i // per_row), int((i % per_row) * 4)
        np.testing.assert_array_equal(sh_tex[row, col0:col0 + 4, :3], sh[i])


def test_splat_sh_degree_construct_below_data():
    positions, covariances, _ = _make_splats(20)
    sh, opacity = _make_sh(20, degree=3)
    v = visuals.GaussianSplatVisual(positions, covariances, sh,
                                    opacities=opacity, sh_degree=1)
    assert v.sh_degree == 1
    assert v.max_sh_degree == 3


def test_splat_sh_degree_invalid_values():
    """A degree above what the data supports, or outside 0..3, is an error."""
    positions, covariances, colors = _make_splats(20)
    with pytest.raises(ValueError, match="only degree-0"):
        visuals.GaussianSplatVisual(positions, covariances, colors,
                                    sh_degree=2)

    sh, opacity = _make_sh(20, degree=3)
    v = visuals.GaussianSplatVisual(positions, covariances, sh,
                                    opacities=opacity)
    for bad in (4, -1):
        with pytest.raises(ValueError, match="0, 1, 2 or 3"):
            v.sh_degree = bad


def test_splat_sh_degree_persists_across_set_data():
    """An explicit degree survives a data update; None goes back to tracking."""
    positions, covariances, colors = _make_splats(20)
    sh, opacity = _make_sh(20, degree=3)
    v = visuals.GaussianSplatVisual(positions, covariances, sh,
                                    opacities=opacity)

    v.sh_degree = 2
    with pytest.raises(ValueError, match="only degree-0"):
        v.set_data(colors=colors)
    assert v.sh_degree == 2      # rejected update leaves the degree alone

    v.sh_degree = None           # unpinned, back to tracking the data
    assert v.sh_degree == 3
    v.set_data(colors=colors)
    assert v.sh_degree == 0


def test_splat_bad_shapes():
    positions, covariances, colors = _make_splats()
    with pytest.raises(ValueError):
        visuals.GaussianSplatVisual(positions[:, :2], covariances, colors)
    with pytest.raises(ValueError):
        visuals.GaussianSplatVisual(positions, covariances[:, 0], colors)
    with pytest.raises(ValueError):
        # (N, 2) is neither RGB nor RGBA
        visuals.GaussianSplatVisual(positions, covariances, colors[:, :2])
    with pytest.raises(ValueError):
        # K = 5 is not a valid spherical-harmonic coefficient count
        visuals.GaussianSplatVisual(positions, covariances,
                                    np.zeros((len(positions), 5, 3), np.float32),
                                    opacities=0.5)


def test_splat_values_out_of_range():
    positions, covariances, colors = _make_splats()
    bad = colors.copy()
    bad[0, 0] = 1.5
    with pytest.raises(ValueError, match="between 0 and 1"):
        visuals.GaussianSplatVisual(positions, covariances, bad)
    bad = colors[:, :3].copy()
    bad[0, 0] = -0.1
    with pytest.raises(ValueError, match="between 0 and 1"):
        visuals.GaussianSplatVisual(positions, covariances, bad, opacities=1.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        visuals.GaussianSplatVisual(positions, covariances, colors[:, :3],
                                    opacities=2.0)


def test_splat_mismatched_lengths():
    positions, covariances, colors = _make_splats(10)
    with pytest.raises(ValueError, match="length"):
        visuals.GaussianSplatVisual(positions, covariances, colors[:5])


def test_splat_empty():
    """An empty visual constructs and contributes no bounds."""
    v = visuals.GaussianSplatVisual(np.zeros((0, 3), np.float32),
                                    np.zeros((0, 3, 3), np.float32),
                                    np.zeros((0, 4), np.float32))
    assert v._compute_bounds(0, None) is None

    # emptying a populated visual drops its bounds too
    positions, covariances, colors = _make_splats()
    v = visuals.GaussianSplatVisual(positions, covariances, colors)
    assert v._compute_bounds(0, None) is not None
    v.set_data(np.zeros((0, 3), np.float32), np.zeros((0, 3, 3), np.float32),
               np.zeros((0, 4), np.float32))
    assert v._compute_bounds(0, None) is None


def test_splat_antialias_property():
    positions, covariances, colors = _make_splats()
    v = visuals.GaussianSplatVisual(positions, covariances, colors)
    assert v.antialias is False
    assert v.shared_program["u_antialias"] == 0.0

    v = visuals.GaussianSplatVisual(positions, covariances, colors,
                                    antialias=True)
    assert v.antialias is True
    assert v.shared_program["u_antialias"] == 1.0
    v.antialias = False
    assert v.shared_program["u_antialias"] == 0.0


@requires_pyopengl()
@requires_application()
def test_splat_draw():
    """A single opaque splat should paint pixels near the screen center."""
    with TestingCanvas(size=(100, 100), bgcolor='black') as c:
        use(gl='gl+')
        positions = np.array([[0, 0, 0]], dtype=np.float32)
        covariances = (np.eye(3, dtype=np.float32) * 0.1)[np.newaxis]
        colors = np.array([[1, 1, 1, 1]], dtype=np.float32)  # opaque white

        view = c.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(fov=0, distance=3.0)
        scene.visuals.GaussianSplat(positions, covariances, colors,
                                    parent=view.scene)
        render = c.render()
        # something was drawn on the (otherwise black) canvas
        assert render[..., :3].sum() > 0


@requires_pyopengl()
@requires_application()
@pytest.mark.parametrize('degree', [0, 1, 2, 3])
def test_splat_draw_sh_matches_plain_color(degree):
    """SH with only a DC term renders the same as the equivalent RGBA color.

    This also compiles the degree-specialized `eval_sh` on a real driver.
    """
    positions = np.array([[0, 0, 0]], dtype=np.float32)
    covariances = (np.eye(3, dtype=np.float32) * 0.1)[np.newaxis]
    rgba = np.array([[0.2, 0.6, 0.9, 1.0]], dtype=np.float32)

    # the DC coefficient that encodes the same color; higher orders are zero,
    # so the result must be view-independent and identical at every degree
    k = (degree + 1) ** 2
    sh = np.zeros((1, k, 3), dtype=np.float32)
    sh[0, 0] = (rgba[0, :3] - 0.5) / 0.28209479177387814

    rendered = []
    for colors, kwargs in [(rgba, {}), (sh, {'opacities': rgba[:, 3]})]:
        with TestingCanvas(size=(100, 100), bgcolor='black') as c:
            use(gl='gl+')
            view = c.central_widget.add_view()
            view.camera = scene.cameras.TurntableCamera(fov=0, distance=3.0)
            scene.visuals.GaussianSplat(positions, covariances, colors,
                                        parent=view.scene, **kwargs)
            rendered.append(c.render())

    assert rendered[0][..., :3].sum() > 0
    np.testing.assert_allclose(rendered[1], rendered[0], atol=1)


@requires_pyopengl()
@requires_application()
def test_splat_draw_after_degree_toggle():
    """Toggling the degree recompiles the program and still draws."""
    n = 50
    positions, covariances, _ = _make_splats(n)
    sh, opacity = _make_sh(n, degree=3)
    with TestingCanvas(size=(100, 100), bgcolor='black') as c:
        use(gl='gl+')
        view = c.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(fov=45)
        splat = scene.visuals.GaussianSplat(positions, covariances, sh,
                                            opacities=opacity, parent=view.scene)
        view.camera.set_range()
        for degree in (3, 0, 2, 1, 3):
            splat.sh_degree = degree
            assert c.render()[..., :3].sum() > 0


@requires_pyopengl()
@requires_application()
def test_splat_draw_antialias_dims_small_splats():
    """Compensating for the dilation lowers the opacity of a small splat."""
    positions = np.array([[0, 0, 0]], dtype=np.float32)
    # about a pixel across, so the dilation roughly doubles its area
    covariances = (np.eye(3, dtype=np.float32) * 1e-5)[np.newaxis]
    colors = np.array([[1, 1, 1, 1]], dtype=np.float32)
    with TestingCanvas(size=(100, 100), bgcolor='black') as c:
        use(gl='gl+')
        view = c.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(fov=0, distance=3.0)
        splat = scene.visuals.GaussianSplat(positions, covariances, colors,
                                            parent=view.scene)
        plain = c.render()[..., :3].astype(float).sum()
        splat.antialias = True
        compensated = c.render()[..., :3].astype(float).sum()
    assert plain > 0
    assert 0 < compensated < plain


@requires_pyopengl()
@requires_application()
def test_splat_slots_address_every_splat():
    """The uploaded draw order names each splat's (column, row) texture slot."""
    n = 200
    positions, covariances, colors = _make_splats(n)
    with TestingCanvas(size=(100, 100), bgcolor='black') as c:
        use(gl='gl+')
        view = c.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(fov=45)
        splat = scene.visuals.GaussianSplat(positions, covariances, colors,
                                            parent=view.scene)
        view.camera.set_range()

        uploaded = []
        splat._slot_vbo.set_data = lambda a, **kw: uploaded.append(a)
        c.render()

        slots = uploaded[-1]
        assert slots.shape == (n, 2)
        # every splat appears exactly once, and each slot decodes to its index
        order = (slots[:, 1] * splat._per_row + slots[:, 0]).astype(np.int64)
        np.testing.assert_array_equal(np.sort(order), np.arange(n))


@requires_pyopengl()
@requires_application()
def test_splat_degenerate_depth_still_draws_every_splat():
    """With no depth to sort by, the order still names every splat, and the
    next real view direction re-sorts."""
    n = 30
    positions, covariances, colors = _make_splats(n)
    with TestingCanvas(size=(100, 100), bgcolor='black') as c:
        use(gl='gl+')
        view = c.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(fov=45)
        splat = scene.visuals.GaussianSplat(positions, covariances, colors,
                                            parent=view.scene)
        view.camera.set_range()

        uploaded = []
        splat._slot_vbo.set_data = lambda a, **kw: uploaded.append(a)
        real_gradient = splat._depth_gradient
        splat._depth_gradient = lambda view: np.zeros(3, np.float32)
        c.render()
        assert len(uploaded) == 1 and uploaded[0].shape == (n, 2)
        c.render()
        assert len(uploaded) == 1   # not re-uploaded while still degenerate

        splat._depth_gradient = real_gradient
        c.render()
        assert len(uploaded) == 2


@requires_pyopengl()
@requires_application()
def test_splat_resort_gating():
    """The depth sort re-runs on rotation but skips pan/zoom/static redraws."""
    positions, covariances, colors = _make_splats()
    with TestingCanvas(size=(100, 100), bgcolor='black') as c:
        use(gl='gl+')
        view = c.central_widget.add_view()
        view.camera = scene.cameras.TurntableCamera(fov=45)
        splat = scene.visuals.GaussianSplat(positions, covariances, colors,
                                            parent=view.scene)
        view.camera.set_range()

        c.render()
        view_dir = splat._last_view_dir.copy()
        assert view_dir is not None

        # static redraw, zoom and pan don't change the back-to-front order
        c.render()
        assert np.array_equal(view_dir, splat._last_view_dir)
        view.camera.scale_factor *= 0.5
        c.render()
        assert np.array_equal(view_dir, splat._last_view_dir)
        view.camera.center = (0.3, 0.0, 0.0)
        c.render()
        assert np.array_equal(view_dir, splat._last_view_dir)

        # a large rotation triggers a re-sort
        view.camera.azimuth += 20.0
        c.render()
        assert not np.array_equal(view_dir, splat._last_view_dir)
        view_dir_rotated = splat._last_view_dir.copy()

        # a color change does not affect the back-to-front order
        splat.set_data(colors=colors * 0.5)
        c.render()
        assert np.array_equal(splat._last_view_dir, view_dir_rotated)

        # new centers do force a re-sort, even from an unmoved camera
        epoch = splat._sort_epoch
        splat.set_data(positions=positions * 2)
        assert splat._sort_epoch != epoch
        c.render()
        assert splat._last_sort_epoch == splat._sort_epoch


@requires_pyopengl()
@requires_application()
def test_splat_two_views_sort_independently():
    """Two views of one visual keep separate draw orders and don't thrash.

    The back-to-front order depends on the camera direction, so sharing one
    order between views would make each draw invalidate the other's sort.
    """
    from vispy.visuals.transforms import MatrixTransform, TransformSystem

    positions, covariances, colors = _make_splats(50)
    with TestingCanvas(size=(100, 100)) as c:
        use(gl='gl+')
        splat = visuals.GaussianSplatVisual(positions, covariances, colors)
        front, back = splat.view(), splat.view()
        for view, angle in ((front, 0), (back, 180)):
            transforms = TransformSystem(c)
            matrix = MatrixTransform()
            matrix.rotate(angle, (0, 1, 0))
            transforms.visual_transform = matrix
            view.transforms = transforms

        # each view owns its index buffer; the splat data stays shared
        assert front._slot_vbo is not back._slot_vbo

        resorts = {id(front): 0, id(back): 0}
        real_sort = visuals.GaussianSplatVisual._sort

        def counting_sort(self, view):
            before = view._last_view_dir
            before = None if before is None else before.copy()
            real_sort(self, view)
            if before is None or not np.array_equal(before, view._last_view_dir):
                resorts[id(view)] += 1

        visuals.GaussianSplatVisual._sort = counting_sort
        try:
            for _ in range(5):
                for view in (front, back):
                    view._prepare_draw(view=view)
        finally:
            visuals.GaussianSplatVisual._sort = real_sort

        # one sort each, not one per draw
        assert resorts == {id(front): 1, id(back): 1}
        # and they really are looking in opposite directions
        assert np.dot(front._last_view_dir, back._last_view_dir) < -0.99

        # new positions invalidate every view's order, not just one
        splat.set_data(positions=positions * 2)
        assert front._last_sort_epoch != splat._sort_epoch
        assert back._last_sort_epoch != splat._sort_epoch
        for view in (front, back):
            view._prepare_draw(view=view)
            assert view._last_sort_epoch == splat._sort_epoch


def test_splat_opacities_not_shadowed_by_node():
    """`Node.opacity` is a whole-visual alpha and must not shadow ours."""
    from vispy import scene

    positions, covariances, _ = _make_splats(10)
    sh, opacity = _make_sh(10, degree=1)
    splat = scene.visuals.GaussianSplat(positions, covariances, sh,
                                        opacities=opacity)
    # reachable on the scene class, which is what the docs and example use
    np.testing.assert_array_equal(splat.opacities, opacity)
    # and Node's own scalar alpha still works, unrelated
    assert splat.opacity == 1.0


def test_splat_capacity_and_overflow_message():
    """Too many splats for the degree is a ValueError naming a degree that fits."""
    from vispy.visuals.gaussian_splat import _capacity, _check_capacity

    # capacity falls as the coefficient count rises, and every degree beats the
    # old float32-index ceiling of 2**24 that the (row, col) slot removed
    caps = {d: _capacity((d + 1) ** 2) for d in range(4)}
    assert caps[0] > caps[1] > caps[2] >= caps[3]
    assert caps[3] >= 2 ** 24

    # inside the limit is silent
    _check_capacity(caps[3], 16)

    # over it names the count, the degree, the limit and a degree that fits
    with pytest.raises(ValueError) as excinfo:
        _check_capacity(caps[3] + 1, 16)
    message = str(excinfo.value)
    assert f"{caps[3] + 1:,}" in message
    assert "degree 3" in message
    assert f"{caps[3]:,}" in message
    assert "sh_degree=2" in message

    # and when nothing fits it says so rather than suggesting an impossible one
    with pytest.raises(ValueError, match="no SH degree fits"):
        _check_capacity(caps[0] + 1, 16)


def test_splat_failed_pack_leaves_visual_unchanged():
    """A capacity error during packing must not half-apply the update."""
    import vispy.visuals.gaussian_splat as gsm

    positions, covariances, _ = _make_splats(50)
    sh, opacity = _make_sh(50, degree=3)
    v = visuals.GaussianSplatVisual(positions, covariances, sh,
                                    opacities=opacity, sh_degree=0)
    before = (v._sh_degree, v._sh_func, v._splat_sh, v._sort_epoch)

    real = gsm._texture_for

    def overflow_on_color(count, texels_per_splat, per_row):
        if texels_per_splat == 16:
            raise ValueError("simulated capacity overflow")
        return real(count, texels_per_splat, per_row)

    gsm._texture_for = overflow_on_color
    try:
        with pytest.raises(ValueError, match="simulated"):
            v.sh_degree = 3
    finally:
        gsm._texture_for = real

    # nothing committed: in particular _sh_func was not swapped for one whose
    # sh_tex_size/sh_per_row never got assigned, which would break every draw
    assert (v._sh_degree, v._sh_func, v._splat_sh, v._sort_epoch) == before
    assert v.sh_degree == 0


run_tests_if_main()

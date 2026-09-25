# -*- coding: utf-8 -*-
# vispy: testskip
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
"""
3D Gaussian Splatting
=====================

Render a 3D Gaussian Splatting scene with the
:class:`~vispy.scene.visuals.GaussianSplat` visual.

Controls:
* 0-3 - set the spherical-harmonic degree used for view-dependent color
* a - toggle anti-aliasing opacity compensation

This reads a ``.ply`` file in the layout emitted by the INRIA "3D Gaussian
Splatting for Real-Time Radiance Field Rendering" code and compatible tools
(x/y/z, f_dc_*, f_rest_*, opacity, scale_*, rot_*), builds a 3D covariance per
Gaussian from the log-space scales and rotation quaternions, and hands the
resulting arrays to the ``GaussianSplat`` visual.

The spherical-harmonic color coefficients (f_dc_* plus the higher-order
f_rest_*) are passed through so the visual can render view-dependent color.
The degree can be changed while the scene is up: dropping to 0 keeps only the
constant (DC) term, which is cheaper to render and uses much less GPU memory.

Scenes trained with anti-aliasing need their opacity compensated for the
visual's screen-space dilation. The ply layout has no standard field for this,
so it is read from the ``postshot.anti_aliasing`` header comment that Postshot
writes (the sample scene has it); use ``--antialias``/``--no-antialias`` to
override, or press ``a`` to toggle it while the scene is up.

With no ply file provided this script fetches a sample file of a clusterfly
from the vispy demo-data repository. This file was orignally created by Dany
Bittel and retrieved from https://superspl.at/scene/285082b2 (licensed CC-BY).

Requires the ``plyfile`` and ``scipy`` packages
(``pip install plyfile scipy``).

Usage::

    python gaussian_splatting.py
    python gaussian_splatting.py path/to/point_cloud.ply
"""
import argparse
import sys

import numpy as np

from vispy import app, scene, use
from vispy.io import load_data_file

# Full gl+ context is required for instanced rendering.
use(gl='gl+')

DEFAULT_PLY = 'gaussian_splatting/cluster/cluster_fly_S.ply'


def load_splats(path):
    """Read a 3DGS ply and return per-Gaussian arrays compatible with
    the GaussianSplat visual.

    Returns ``(positions, covariances, sh_coeffs, opacity, antialias)``. The
    first four are float32 arrays with shapes (N, 3), (N, 3, 3), (N, K, 3) and
    (N,), where ``sh_coeffs`` are the RGB spherical-harmonic coefficients
    (``sh_coeffs[:, 0]`` is the DC term) and ``opacity`` is the per-Gaussian
    peak alpha. ``K = (degree+1)**2`` is set by the coefficients present in
    the file. ``antialias`` says whether the header marks the scene as trained
    with anti-aliasing.
    """
    from plyfile import PlyData
    from scipy.spatial.transform import Rotation

    ply = PlyData.read(path)
    v = ply.elements[0].data
    # plyfile files a comment under the element it follows, and Postshot
    # writes this one after the vertex element line
    comments = ply.comments + [c for el in ply.elements for c in el.comments]
    antialias = 'postshot.anti_aliasing=1' in comments

    xyz = np.stack([v['x'], v['y'], v['z']], axis=-1)

    # scale_* are log-space std-devs
    scales = np.exp(np.stack([v[f'scale_{i}'] for i in range(3)], axis=-1))

    # rot_* is a (w, x, y, z) quaternion
    quats = np.stack([v[f'rot_{i}'] for i in range(4)], axis=-1)
    # scipy uses scalar-last (x, y, z, w) order and normalizes internally
    R = Rotation.from_quat(quats[:, [1, 2, 3, 0]]).as_matrix()

    # Sigma = R S S^T R^T
    # M = R @ diag(scales), so
    # Sigma = M @ M^T
    # (float64 throughout: scipy's as_matrix always returns float64)
    M = R * scales[:, None, :]
    # for each splat (row), do M @ M.T
    sigma = M @ M.swapaxes(-1, -2)

    n = len(xyz)

    # SH color coefficients: the DC term (f_dc_*) is coefficient 0, the
    # view-dependent detail lives in the f_rest_* fields. The ply stores
    # f_rest channel-major: [ch0_coeff1.., ch1_coeff1.., ch2_coeff1..], i.e.
    # f_rest_{c*(K-1)+j} is coefficient (j+1) of channel c.
    names = v.dtype.names
    n_rest = sum(name.startswith('f_rest_') for name in names)
    k = n_rest // 3 + 1                            # coeffs per channel

    sh = np.zeros((n, k, 3), dtype=np.float32)
    sh[:, 0, :] = np.stack([v[f'f_dc_{c}'] for c in range(3)], axis=-1)
    if k > 1:
        rest = np.stack([v[f'f_rest_{i}'] for i in range(n_rest)], axis=-1)
        sh[:, 1:, :] = rest.reshape(n, 3, k - 1).transpose(0, 2, 1)

    opacity = 1.0 / (1.0 + np.exp(-v['opacity']))    # stored as a logit

    f32 = np.float32
    return (
        np.ascontiguousarray(xyz, f32),
        np.ascontiguousarray(sigma, f32),
        np.ascontiguousarray(sh, f32),
        np.ascontiguousarray(opacity, f32),
        antialias,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('ply', nargs='?', default=None,
                        help='path to a 3DGS point_cloud.ply '
                             '(default fetches a sample)')
    parser.add_argument('--up', default='+z',
                        choices=['+x', '-x', '+y', '-y', '+z', '-z'],
                        help="scene up-axis. If the scene is upside down use "
                             "--up=-z; note negative axes need the '=' form. "
                             "Then try +y/-y (default +z)")
    parser.add_argument('--antialias', default=None,
                        action=argparse.BooleanOptionalAction,
                        help='compensate opacity for scenes trained with '
                             'anti-aliasing (default: read from the ply '
                             'header)')
    args = parser.parse_args()

    if args.ply is not None:
        path = args.ply
        up = args.up
    else:
        path = load_data_file(DEFAULT_PLY)
        up = "-y"

    positions, covariances, sh_coeffs, opacity, antialias = load_splats(path)
    if args.antialias is not None:
        antialias = args.antialias

    canvas = scene.SceneCanvas(keys='interactive', show=True, bgcolor='black')
    view = canvas.central_widget.add_view()
    view.camera = scene.cameras.TurntableCamera(fov=45.0, up=up)

    # sh_coeffs is (N, K, 3), so it is taken as spherical harmonics; a plain
    # (N, 4) RGBA array would be accepted here just as well
    splats = scene.visuals.GaussianSplat(positions, covariances, sh_coeffs,
                                         opacities=opacity,
                                         antialias=antialias,
                                         parent=view.scene)
    view.camera.set_range()

    print(f'loaded {len(positions):,} gaussians, '
          f'rendering at SH degree {splats.sh_degree}')
    print(f'press 0-{splats.max_sh_degree} to change the spherical-harmonic '
          'degree used for view-dependent color')
    print(f'anti-aliasing {"on" if antialias else "off"}, press a to toggle')

    @canvas.events.key_press.connect
    def on_key_press(event):
        """0-3 switch the rendered SH degree. The coefficients are kept host
        side, so this only changes how many of them reach the GPU and how much
        of the basis the shader evaluates. a toggles anti-aliasing."""
        if event.text == 'a':
            splats.antialias = not splats.antialias
            print(f'anti-aliasing {"on" if splats.antialias else "off"}')
            return
        if event.text in ('0', '1', '2', '3'):
            degree = int(event.text)
            if degree > splats.max_sh_degree:
                print(f'this scene only has degree-{splats.max_sh_degree} '
                      'coefficients')
                return
            splats.sh_degree = degree
            print(f'SH degree {degree}')

    canvas.show()
    if sys.flags.interactive != 1:
        app.run()


if __name__ == '__main__':
    main()

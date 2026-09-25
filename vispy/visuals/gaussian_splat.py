# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) Vispy Development Team. All Rights Reserved.
# Distributed under the (new) BSD License. See LICENSE.txt for more info.
# -----------------------------------------------------------------------------
"""A visual for rendering 3D Gaussian Splatting scenes."""

import numpy as np

from .. import gloo
from ..color import ColorArray
from ..util import logger
from .shaders import Function
from .visual import Visual

# Per-splat data lives in two RGBA32F textures, fetched in the vertex shader by
# splat index; only the sorted index order is re-uploaded per frame.
#
#   geometry, 3 texels/splat: (center.xyz, opacity),
#                             (S00, S01, S02, S11), (S12, S22, _, _)
#   color, K texels/splat:    one spherical-harmonic coefficient per texel, RGB
#                             in .xyz, K = (degree+1)**2 in {1, 4, 9, 16}
#
# Splitting them means a position or covariance change re-uploads only the
# geometry texture. A degree change re-uploads the color texture and, because
# splats-per-row is shared, relayouts the (much smaller) geometry one too.
#
# Rows hold whole splats, so a splat's texels are a linear span within one row,
# and BOTH textures use the same splats-per-row. That lets the draw order be
# uploaded as the (row, column) slot itself rather than a running index: a
# single float32 index would only stay exact to 2**24, which would cap the
# visual well below what the textures can actually hold.
#
# Capacity is therefore splats-per-row x _MAX_TEX_HEIGHT, and splats-per-row is
# the row width divided by the widest record (max(_GEOM_TEXELS, K)).
_GEOM_TEXELS = 3
_MAX_TEX_SIZE = 16384
_MAX_TEX_HEIGHT = _MAX_TEX_SIZE

# degree-0 spherical-harmonics constant, converts plain RGB <-> the DC term
_SH_C0 = 0.28209479177387814

# lets set_data tell `sh_degree=None` (unpin) from an omitted argument
_UNSET = object()

# origin plus one unit step per axis, used to recover the depth gradient
_DEPTH_PROBES = np.array(
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
)


def _degree_of(sh):
    """The spherical-harmonic degree of an (N, K, 3) coefficient array."""
    return int(round(sh.shape[1] ** 0.5)) - 1


def _splats_per_row(sh_count):
    """Splats per texture row, shared by the geometry and color textures.

    Both are indexed by the same (row, column) slot, so the row has to hold a
    whole number of splats in each -- the widest record decides.
    """
    return max(1, _MAX_TEX_SIZE // max(_GEOM_TEXELS, sh_count))


def _capacity(sh_count):
    """How many splats fit at an SH coefficient count."""
    return _splats_per_row(sh_count) * _MAX_TEX_HEIGHT


def _check_capacity(count, sh_count):
    """Raise if ``count`` splats cannot be addressed at this SH degree."""
    if count <= _capacity(sh_count):
        return
    degree = int(round(sh_count ** 0.5)) - 1
    fits = [d for d in range(degree) if count <= _capacity((d + 1) ** 2)]
    advice = (f"; sh_degree={max(fits)} would fit" if fits else
              "; no SH degree fits this many splats")
    raise ValueError(
        f"{count:,} splats exceed what this visual can address at SH "
        f"degree {degree}: the limit is {_capacity(sh_count):,} "
        f"({_splats_per_row(sh_count)} splats per row x {_MAX_TEX_HEIGHT} "
        f"rows){advice}"
    )


def _texture_for(count, texels_per_splat, per_row):
    """Allocate a data texture and a writable per-splat record view into it.

    Returns ``(texture, records)``, where ``records`` is a
    (count, texels_per_splat, 4) *view* onto ``texture`` -- rows hold `per_row`
    whole splats, so splat i lands at slot (i // per_row, i % per_row).
    Callers fill ``records`` in place; the padding stays zero.
    """
    width = per_row * texels_per_splat
    height = int(np.ceil(count / per_row)) if count else 1
    tex = np.zeros((height, width, 4), np.float32)
    return tex, tex.reshape(-1, texels_per_splat, 4)[:count]


def _parse_positions(positions):
    pos = np.ascontiguousarray(positions, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")
    return pos


def _parse_covariances(covariances):
    """Split symmetric 3x3s into (S00, S01, S02) and (S11, S12, S22)."""
    cov = np.asarray(covariances, dtype=np.float32)
    if cov.ndim != 3 or cov.shape[1:] != (3, 3):
        raise ValueError(
            f"covariances must have shape (N, 3, 3), got {cov.shape}"
        )
    return np.ascontiguousarray(cov[:, 0, :]), np.ascontiguousarray(
        np.stack([cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], axis=-1)
    )


def _check_unit_range(arr, name):
    """Raise unless every value is in [0, 1], as ColorArray requires."""
    if arr.size and (arr.min() < 0 or arr.max() > 1):
        raise ValueError(f"{name} values must be between 0 and 1")


def _parse_colors(colors, count):
    """Validate `colors` and return (SH coefficients, alpha or None).

    ``count`` is used only to broadcast a single color over every splat.
    """
    if isinstance(colors, str) or np.ndim(colors) <= 1:
        # one color for all splats, e.g. 'white', '#ff000080' or (1, 0, 0)
        colors = np.broadcast_to(ColorArray(colors).rgba, (count, 4))
    arr = np.asarray(colors, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[1] in (3, 4):
        _check_unit_range(arr, "colors")
        # a plain color is the degree-0 SH case: dc = (rgb - 0.5) / C0
        sh = np.ascontiguousarray(((arr[:, :3] - 0.5) / _SH_C0)[:, None, :])
        alpha = np.ascontiguousarray(arr[:, 3]) if arr.shape[1] == 4 else None
        return sh, alpha
    if arr.ndim == 3 and arr.shape[2] == 3 and arr.shape[1] in (1, 4, 9, 16):
        return np.ascontiguousarray(arr), None
    raise ValueError(
        "colors must be a single color, or have shape (N, 3) RGB, (N, 4) "
        "RGBA, or (N, K, 3) spherical-harmonic coefficients with K in "
        f"{{1, 4, 9, 16}}; got shape {arr.shape}"
    )


def _parse_opacities(opacities, count):
    arr = np.asarray(opacities, dtype=np.float32)
    if arr.ndim == 0:
        arr = np.broadcast_to(arr, (count,))
    elif arr.ndim != 1:
        raise ValueError(
            f"opacities must be a scalar or have shape (N,), got {arr.shape}"
        )
    _check_unit_range(arr, "opacities")
    return np.ascontiguousarray(arr)


def _parse_sh_degree(degree):
    if degree is None:
        return None
    degree = int(degree)
    if not 0 <= degree <= 3:
        raise ValueError(f"sh_degree must be 0, 1, 2 or 3, got {degree}")
    return degree


def _resolve_sh_degree(pinned, max_degree):
    """The degree to render, given the pinned request and what the data allows."""
    if pinned is None:
        return max_degree
    if pinned > max_degree:
        raise ValueError(
            f"sh_degree is set to {pinned}, but only degree-{max_degree} "
            "coefficients were supplied; pass more coefficients or set "
            "sh_degree lower (or to None to track the supplied degree)"
        )
    return pinned


def _check_lengths(pos, cov_a, sh, alpha):
    counts = {"covariances": len(cov_a), "colors": len(sh),
              "opacities": len(alpha)}
    bad = [f"{k} has length {v}" for k, v in counts.items() if v != len(pos)]
    if bad:
        raise ValueError(
            f"positions has length {len(pos)}, but " + ", ".join(bad)
        )


# Spherical-harmonics evaluation, generated per degree so the coefficient count
# is a compile-time constant: fetches are unrolled and no registers are held for
# coefficients this degree does not use. At degree 0 the `dir` argument goes
# unused, so the driver also eliminates view_direction() and its transforms from
# main(), leaving a program as cheap as one with no SH support at all.
_SH_PREAMBLE = """
vec3 eval_sh(vec2 slot, vec3 dir) {
    const float SH_C0 = 0.28209479177387814;
    float row = slot.y;
    float col0 = slot.x * %(k)d.0;
    vec2 inv_size = 1.0 / $sh_tex_size;
"""

_SH_FETCH = ("    vec3 sh%(k)d = texture2D($sh_tex, "
             "(vec2(col0 + %(k)d.0, row) + 0.5) * inv_size).xyz;\n")

_SH_DEGREE_1 = """
    const float SH_C1 = 0.4886025119029199;
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    c += SH_C1 * (-y * sh1 + z * sh2 - x * sh3);
"""

_SH_DEGREE_2 = """
    const float SH_C2_0 = 1.0925484305920792;
    const float SH_C2_1 = -1.0925484305920792;
    const float SH_C2_2 = 0.31539156525252005;
    const float SH_C2_3 = -1.0925484305920792;
    const float SH_C2_4 = 0.5462742152960396;
    float xx = x * x;
    float yy = y * y;
    float zz = z * z;
    float xy = x * y;
    float yz = y * z;
    float xz = x * z;
    c += SH_C2_0 * xy * sh4
       + SH_C2_1 * yz * sh5
       + SH_C2_2 * (2.0 * zz - xx - yy) * sh6
       + SH_C2_3 * xz * sh7
       + SH_C2_4 * (xx - yy) * sh8;
"""

_SH_DEGREE_3 = """
    const float SH_C3_0 = -0.5900435899266435;
    const float SH_C3_1 = 2.890611442640554;
    const float SH_C3_2 = -0.4570457994644658;
    const float SH_C3_3 = 0.3731763325901154;
    const float SH_C3_4 = -0.4570457994644658;
    const float SH_C3_5 = 1.445305721320277;
    const float SH_C3_6 = -0.5900435899266435;
    c += SH_C3_0 * y * (3.0 * xx - yy) * sh9
       + SH_C3_1 * xy * z * sh10
       + SH_C3_2 * y * (4.0 * zz - xx - yy) * sh11
       + SH_C3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh12
       + SH_C3_4 * x * (4.0 * zz - xx - yy) * sh13
       + SH_C3_5 * z * (xx - yy) * sh14
       + SH_C3_6 * x * (xx - 3.0 * yy) * sh15;
"""


def _sh_shader(degree):
    """GLSL source for ``eval_sh`` specialized to a spherical-harmonic degree."""
    k = (degree + 1) ** 2
    src = [_SH_PREAMBLE % {"k": k}]
    src += [_SH_FETCH % {"k": i} for i in range(k)]
    src.append("    vec3 c = SH_C0 * sh0;\n")
    src += [_SH_DEGREE_1, _SH_DEGREE_2, _SH_DEGREE_3][:degree]
    # the SH basis is signed and centered on 0.5; clamp away negative lobes
    src.append("    return max(c + 0.5, 0.0);\n}\n")
    return "".join(src)


VERTEX_SHADER = """
attribute vec2 a_quad;      // per-vertex: quad corner in [-1, 1]
attribute vec2 a_slot;      // per-instance: (column, row) of the splat to draw

uniform sampler2D u_geom;      // static per-splat geometry (RGBA32F)
uniform vec2 u_geom_size;      // geometry texture size in texels (width, height)

uniform float u_eps;        // finite-difference step (visual units)
uniform float u_antialias;  // 1.0 to compensate opacity for the dilation

const float c_cutoff = 3.0;    // quad half-extent in sigmas
const float c_dilation = 0.3;  // low-pass dilation added to screen cov (px^2)

varying vec4 v_color;
varying vec2 v_offset;      // pixel offset from center at this vertex
varying vec3 v_conic;       // inverse screen covariance: c00, c01, c11

vec4 fetch_geom(float col, float row) {
    // nearest sampling; +0.5 centers the sample on the texel
    vec2 uv = (vec2(col, row) + 0.5) / u_geom_size;
    return texture2D(u_geom, uv);
}

vec2 project(vec3 p) {
    vec4 fb = $visual_to_framebuffer(vec4(p, 1.0));
    return fb.xy / fb.w;
}

// direction from the camera through p, in visual coordinates. Recovered
// purely from the projection (works for any camera): p projects to a point
// on the near and far clip planes; the line between them is the view ray.
vec3 view_direction(vec3 p) {
    vec4 clip = $visual_to_document(vec4(p, 1.0));
    vec4 near_clip = clip; near_clip.z = -clip.w;
    vec4 far_clip = clip;  far_clip.z = clip.w;
    vec4 nh = $document_to_visual(near_clip);
    vec4 fh = $document_to_visual(far_clip);
    return normalize(fh.xyz / fh.w - nh.xyz / nh.w);
}

void main() {
    // the slot is uploaded already split into (column, row), so there is no
    // per-vertex division and no index too large to hold exactly in a float
    float row = a_slot.y;
    float col0 = a_slot.x * 3.0;  // _GEOM_TEXELS
    vec4 g0 = fetch_geom(col0, row);
    vec4 g1 = fetch_geom(col0 + 1.0, row);
    vec4 g2 = fetch_geom(col0 + 2.0, row);

    vec3 a_center = g0.xyz;
    float a_alpha = g0.w;
    vec3 a_cov_a = vec3(g1.x, g1.y, g1.z);   // Sigma00, Sigma01, Sigma02
    vec3 a_cov_b = vec3(g1.w, g2.x, g2.y);   // Sigma11, Sigma12, Sigma22

    vec3 rgb = $eval_sh(a_slot, view_direction(a_center));
    vec4 a_color = vec4(rgb, a_alpha);

    vec4 fb_center = $visual_to_framebuffer(vec4(a_center, 1.0));
    vec2 c = fb_center.xy / fb_center.w;

    // finite-difference Jacobian J (2x3)
    // get visual-space displacement -> pixels for each dim
    vec2 jx = (project(a_center + vec3(u_eps, 0.0, 0.0)) - c) / u_eps;
    vec2 jy = (project(a_center + vec3(0.0, u_eps, 0.0)) - c) / u_eps;
    vec2 jz = (project(a_center + vec3(0.0, 0.0, u_eps)) - c) / u_eps;

    mat3 sigma = mat3(
        a_cov_a.x, a_cov_a.y, a_cov_a.z,
        a_cov_a.y, a_cov_b.x, a_cov_b.y,
        a_cov_a.z, a_cov_b.y, a_cov_b.z
    );

    // screen covariance Sigma2 = J Sigma J^T (2x2)
    // r0, r1 are rows of J
    vec3 r0 = vec3(jx.x, jy.x, jz.x);
    vec3 r1 = vec3(jx.y, jy.y, jz.y);
    vec3 sr0 = sigma * r0;
    vec3 sr1 = sigma * r1;
    float ca0 = dot(r0, sr0);               // Sigma2_00 before dilation
    float cb = dot(r0, sr1);                // Sigma2_01
    float cc0 = dot(r1, sr1);               // Sigma2_11 before dilation
    float ca = ca0 + c_dilation;
    float cc = cc0 + c_dilation;

    float det = ca * cc - cb * cb;
    if (det <= 0.0) {
        // degenerate splat
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        return;
    }

    // inverse screen covariance (conic), passed to the fragment shader
    v_conic = vec3(cc, -cb, ca) / det;

    // eigen-decomposition of the symmetric 2x2 to size/orient the quad
    float tr = ca + cc;
    float disc = sqrt(max(tr * tr * 0.25 - det, 0.0));
    float l1 = tr * 0.5 + disc;
    float l2 = max(tr * 0.5 - disc, 0.0);
    vec2 e1;
    if (abs(cb) < 1e-9) {
        e1 = (ca >= cc) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
    } else {
        e1 = normalize(vec2(l1 - cc, cb));
    }
    vec2 e2 = vec2(-e1.y, e1.x);
    float r_major = c_cutoff * sqrt(l1);
    float r_minor = c_cutoff * sqrt(l2);

    vec2 offset = a_quad.x * r_major * e1 + a_quad.y * r_minor * e2;  // pixels

    v_offset = offset;
    v_color = a_color;
    if (u_antialias > 0.5) {
        // scenes trained with anti-aliasing (Mip-Splatting style) expect the
        // dilation to conserve each splat's total weight, so scale the peak
        // opacity by the ratio of the areas before and after dilating
        float det0 = max(ca0 * cc0 - cb * cb, 0.0);
        v_color.a *= sqrt(det0 / det);
    }

    // offset is in true (post-divide) pixels, but framebuffer coords are
    // pre-perspective-divide, so scale by w
    // the GPU's later /w then yields the intended pixel offset
    // no-op under orthographic (w == 1)
    vec4 fb = fb_center;
    fb.xy += offset * fb.w;
    gl_Position = $framebuffer_to_render(fb);
}
"""

FRAGMENT_SHADER = """
varying vec4 v_color;
varying vec2 v_offset;
varying vec3 v_conic;

void main() {
    vec2 o = v_offset;
    float power = -0.5 * (
        v_conic.x * o.x * o.x
        + 2.0 * v_conic.y * o.x * o.y
        + v_conic.z * o.y * o.y
    );

    if (power > 0.0)
        discard;

    float alpha = v_color.a * exp(power);

    if (alpha < 1.0 / 255.0)
        discard;

    // premultiplied "over" blending
    gl_FragColor = vec4(v_color.rgb * alpha, alpha);
}
"""


class GaussianSplatVisual(Visual):
    """Renderer for a 3D Gaussian Splatting scene.

    Each Gaussian ("splat") is defined by a center, a 3x3 covariance matrix,
    and a color. Each is drawn as an instanced screen-aligned (billboarded)
    quad - similar to the instanced Markers visual.

    Color is stored as spherical-harmonic (SH) coefficients, of which a plain
    RGB color is the degree-0 case; ``colors`` accepts either form. Above
    degree 0 the color is *view dependent*, evaluated in the vertex shader from
    the camera-to-Gaussian direction, as stored by tools emitting the INRIA 3D
    Gaussian Splatting ``.ply`` layout.

    The 3D covariance is projected to a 2D screen-space covariance in the
    vertex shader using a finite-difference Jacobian of the visual->framebuffer
    projection (vispy's equivalent of the model-view-projection); finite
    differences because that projection is nonlinear under a perspective
    camera. The fragment shader evaluates the resulting 2D Gaussian. Splats are
    sorted and rendered back-to-front and blended (no depth testing).

    Parameters
    ----------
    positions : (N, 3) array
        Gaussian centers, in visual coordinates.
    covariances : (N, 3, 3) array
        Symmetric positive-definite 3D covariance matrix for each Gaussian.
    colors : array or Color
        One of:

        * a single color for every splat -- anything ``Color`` accepts, such
          as ``'white'``, ``'#ff000080'`` or ``(1, 0, 0)``;
        * a per-splat plain color, as (N, 3) RGB or (N, 4) RGBA in [0, 1];
        * per-splat spherical-harmonic RGB coefficients, as (N, K, 3) where
          ``K = (degree + 1)**2`` is 1, 4, 9 or 16 (degree 0..3) and
          ``colors[:, 0]`` is the DC (view-independent) term.
    opacities : (N,) array or float, optional
        Per-Gaussian peak opacity in [0, 1]. Required unless ``colors`` is
        given as (N, 4) RGBA, which supplies it as the alpha channel. A single
        color also carries an alpha (opaque unless the color says otherwise),
        which ``opacities`` overrides. Named in the plural because the scene
        graph's ``Node.opacity`` is a separate, whole-visual alpha.
    sh_degree : int, optional
        The spherical-harmonic degree to render. ``None`` (the default) renders
        every coefficient supplied; see the ``sh_degree`` property.
    antialias : bool
        Compensate opacity for the screen-space low-pass dilation, as expected
        by scenes trained with anti-aliasing (e.g. Mip-Splatting or gsplat's
        ``"antialiased"`` mode). Leave off (the default) for scenes trained
        like the original 3DGS, which would otherwise render too faint. See
        the ``antialias`` property.

    Notes
    -----
    Per-splat records are stored once in data textures and fetched in the
    vertex shader by index. Splats are depth-sorted on the CPU, but only the
    (N,) index order is re-uploaded, and only when the view *direction* rotates
    (pan, zoom and static redraws reuse the existing order) -- so the heavy
    per-splat data never moves after upload, which keeps interaction smooth for
    up to a few million splats. View-dependent color is free every frame
    because it is evaluated on the GPU.

    The draw order is uploaded as each splat's ``(column, row)`` texture slot
    rather than a running index, so the count is limited by texture area alone
    rather than by float32 index precision. The ceiling falls as the rendered
    SH degree rises -- roughly 89M splats at degree 0 down to 16M at degree 3 --
    and exceeding it raises ``ValueError`` naming a degree that would fit. In
    practice GPU memory binds first: at degree 3 a splat costs 256 bytes of
    color texture plus 48 of geometry.
    """

    def __init__(self, positions, covariances, colors, opacities=None, *,
                 sh_degree=None, antialias=False):
        Visual.__init__(self, VERTEX_SHADER, FRAGMENT_SHADER)

        self._splat_pos = None
        self._splat_cov_a = None
        self._splat_cov_b = None
        self._splat_sh = None       # (N, K, 3) SH coefficients (RGB), as given
        self._splat_alpha = None    # (N,) peak opacity

        # the degree explicitly requested by the user, or None to track
        # max_sh_degree; _sh_degree is the resolved degree actually rendered
        self._sh_degree_pinned = None
        self._sh_degree = None
        self._sh_func = None
        # splats per texture row, shared by both textures (see _splats_per_row)
        self._per_row = None

        # cached bounding box, rows (min, max), for _compute_bounds
        self._bounds = None

        # the real GL_MAX_TEXTURE_SIZE is only readable once a context exists,
        # so the textures are checked against it on the next draw after upload
        self._checked_texture_size = False

        # instancing quad, drawn as a triangle strip
        quad = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32)
        self.shared_program["a_quad"] = gloo.VertexBuffer(quad)

        # bumped whenever the centers change, so each view can tell that its
        # cached draw order went stale without anyone pushing into it (_sort)
        self._sort_epoch = 0

        # the visual is its own first view (see _init_view)
        self._init_view(self)

        # static per-splat data textures (sizes set in _pack_geometry/_pack_sh)
        self._geom_tex = self._make_texture()
        self._sh_tex = self._make_texture()
        self.shared_program["u_geom"] = self._geom_tex

        self._draw_mode = "triangle_strip"

        # premultiplied "over" blending, sort back-to-front on the CPU (_sort)
        self.set_gl_state(
            depth_test=False,
            cull_face=False,
            blend=True,
            blend_func=("one", "one_minus_src_alpha"),
        )

        self.antialias = antialias
        self.set_data(positions, covariances, colors=colors,
                      opacities=opacities,
                      sh_degree=sh_degree)

    def _init_view(self, view):
        """Give ``view`` its own draw order.

        The back-to-front order is a function of the camera direction, so it
        is per-view state: two views of one visual sort differently, and
        sharing one order would make them fight over it and re-sort on every
        draw. Only the (N,) index buffer is per-view -- the heavy per-splat
        textures stay shared, so a second view costs 8 bytes per splat.
        """
        view._slot_vbo = gloo.VertexBuffer(
            np.zeros((1, 2), np.float32), divisor=1)
        # view direction (normalized framebuffer-depth gradient) and data epoch
        # this view's order was built for; the order only goes stale when the
        # direction rotates or the centers change, so _sort skips pan, zoom and
        # static redraws (see _sort). -1 never matches, so a new view sorts once.
        view._last_view_dir = None
        view._last_sort_epoch = -1
        view.view_program["a_slot"] = view._slot_vbo

    def view(self):
        """Return a new view of this visual, with its own draw order."""
        view = Visual.view(self)
        self._init_view(view)
        return view

    @staticmethod
    def _make_texture():
        return gloo.Texture2D(
            np.zeros((1, 1, 4), np.float32),
            interpolation="nearest",
            internalformat="rgba32f",
        )

    @property
    def positions(self):
        """The (N, 3) array of Gaussian centers."""
        return self._splat_pos

    @property
    def covariances(self):
        """The (N, 3, 3) array of per-Gaussian 3D covariance matrices."""
        a, b = self._splat_cov_a, self._splat_cov_b
        sigma = np.zeros((len(a), 3, 3), dtype=np.float32)
        # cov_a = (S00, S01, S02), cov_b = (S11, S12, S22).
        sigma[:, 0, 0], sigma[:, 0, 1], sigma[:, 0, 2] = a[:, 0], a[:, 1], a[:, 2]
        sigma[:, 1, 1], sigma[:, 1, 2], sigma[:, 2, 2] = b[:, 0], b[:, 1], b[:, 2]
        # mirror to the lower triangle
        sigma[:, 1, 0], sigma[:, 2, 0], sigma[:, 2, 1] = a[:, 1], a[:, 2], b[:, 1]
        return sigma

    @property
    def max_sh_degree(self):
        """The spherical-harmonic degree of the coefficients supplied."""
        return _degree_of(self._splat_sh)

    @property
    def sh_degree(self):
        """The spherical-harmonic degree currently being rendered.

        Assigning a lower degree drops the view-dependent detail, which
        shrinks the color texture and simplifies the shader - a runtime
        quality/performance knob. The full set of coefficients is kept host
        side, so the change is reversible.

        Assigning a degree above `max_sh_degree` raises ``ValueError``: there
        is no data to render it from. Assigning ``None`` restores the default
        of tracking `max_sh_degree`, so that new coefficients passed to
        `set_data` are rendered in full; an explicitly assigned degree instead
        persists across `set_data` calls.
        """
        return self._sh_degree

    @sh_degree.setter
    def sh_degree(self, degree):
        self.set_data(sh_degree=degree)

    @property
    def sh_coeffs(self):
        """The (N, K, 3) array of per-Gaussian SH coefficients (RGB).

        This is the full set as supplied, whatever `sh_degree` is rendering.
        """
        return self._splat_sh

    @property
    def opacities(self):
        """The (N,) array of per-Gaussian peak opacities.

        Not ``opacity``: the scene graph's ``Node.opacity`` is a separate,
        whole-visual alpha and would shadow this name on
        ``scene.visuals.GaussianSplat``.
        """
        return self._splat_alpha

    @property
    def colors(self):
        """The per-Gaussian color.

        (N, 4) RGBA when `max_sh_degree` is 0, otherwise the (N, K, 3) SH
        coefficients. Either form is accepted back by `set_data`.
        """
        if self.max_sh_degree == 0:
            # clip away float32 round-trip error, so the result is always
            # valid input to set_data
            rgb = np.clip(0.5 + _SH_C0 * self._splat_sh[:, 0, :], 0.0, 1.0)
            return np.concatenate([rgb, self._splat_alpha[:, None]], axis=-1)
        return self._splat_sh

    @property
    def antialias(self):
        """Whether opacity is compensated for the low-pass dilation.

        Every splat's screen-space covariance is dilated by a small fixed
        amount so that sub-pixel splats don't alias. Scenes trained with
        anti-aliasing expect that dilation to leave each splat's total weight
        unchanged, i.e. its peak opacity scaled down by
        ``sqrt(det(cov2d) / det(cov2d + dilation))``; scenes trained like the
        original 3DGS expect the opacity as stored. The training mode is not
        part of the standard ply layout, though some tools record it (Postshot
        writes a ``postshot.anti_aliasing=1`` header comment).
        """
        return self._antialias

    @antialias.setter
    def antialias(self, value):
        self._antialias = bool(value)
        self.shared_program["u_antialias"] = float(self._antialias)
        self.update()

    def set_data(self, positions=None, covariances=None, colors=None,
                 opacities=None, *, sh_degree=_UNSET):
        """Update any subset of the per-Gaussian data.

        Parameters not supplied are left unchanged. See the class docstring
        for the expected shapes, and `sh_degree` for the degree semantics.
        Everything is validated before any of it is committed, so a call that
        raises leaves the visual as it was.
        """
        self._upload_data(self._prepare_data(
            positions, covariances, colors, opacities, sh_degree))
        self.update()

    def _prepare_data(self, positions, covariances, colors, opacities,
                      sh_degree):
        """Validate the arguments and return the full state to commit."""
        if (opacities is not None and colors is not None
                and np.ndim(colors) == 2 and np.shape(colors)[-1] == 4):
            raise ValueError(
                "opacities is already given by the alpha channel of "
                "(N, 4) RGBA colors; pass (N, 3) RGB to set it separately"
            )

        # unsupplied fields fall back to what is already held
        pos = self._splat_pos if positions is None else _parse_positions(positions)
        cov_a, cov_b = ((self._splat_cov_a, self._splat_cov_b)
                        if covariances is None else _parse_covariances(covariances))
        sh, alpha = self._splat_sh, self._splat_alpha
        rgba_alpha = None
        if colors is not None:
            sh, rgba_alpha = _parse_colors(colors, len(pos))
            alpha = alpha if rgba_alpha is None else rgba_alpha

        if alpha is None and opacities is None:
            raise ValueError(
                "opacities is required unless colors is a single color "
                "or (N, 4) RGBA; pass it alongside (N, 3) RGB or (N, K, 3) "
                "SH colors"
            )
        if opacities is not None:
            alpha = _parse_opacities(opacities, len(pos))
        _check_lengths(pos, cov_a, sh, alpha)

        pinned = (self._sh_degree_pinned if sh_degree is _UNSET
                  else _parse_sh_degree(sh_degree))
        degree = _resolve_sh_degree(pinned, _degree_of(sh))
        # capacity depends on the degree actually being rendered, so it can
        # only be checked once the degree is resolved
        sh_count = (degree + 1) ** 2
        _check_capacity(len(pos), sh_count)

        alpha_given = opacities is not None or rgba_alpha is not None
        degree_changed = degree != self._sh_degree
        per_row = _splats_per_row(sh_count)
        return dict(
            per_row=per_row,
            pos=pos, cov_a=cov_a, cov_b=cov_b, sh=sh, alpha=alpha,
            pinned=pinned, degree=degree,
            moved=positions is not None,
            # an argument that was supplied counts as dirty even if it is the
            # same array as before -- it may have been mutated in place
            # a degree change can move splats-per-row, which relayouts the
            # geometry texture too even though its own contents did not change
            per_row_changed=per_row != self._per_row,
            geom_dirty=(positions is not None or covariances is not None
                        or alpha_given or per_row != self._per_row),
            degree_changed=degree_changed,
            sh_dirty=colors is not None or degree_changed,
        )

    def _upload_data(self, data):
        """Commit validated state and refresh whatever it invalidated.

        Packing runs first: `_texture_for` can still raise on a capacity
        overflow, and committing before that would leave the shader pointing
        at coefficients that were never uploaded.
        """
        # keeping geometry and color in separate textures means a position or
        # covariance change skips the (much larger) color pack
        geom = self._pack_geometry(data) if data["geom_dirty"] else None
        color = self._pack_sh(data) if data["sh_dirty"] else None

        # ---- past here nothing can fail ----
        self._splat_pos = data["pos"]
        self._splat_cov_a, self._splat_cov_b = data["cov_a"], data["cov_b"]
        self._splat_sh, self._splat_alpha = data["sh"], data["alpha"]
        self._sh_degree_pinned = data["pinned"]
        self._per_row = data["per_row"]

        if data["moved"] and len(self._splat_pos):
            lo, hi = self._splat_pos.min(0), self._splat_pos.max(0)
            self._bounds = np.array([lo, hi], dtype=np.float32)
            # ~1% of the largest bounding-box side keeps the finite-difference
            # step well-conditioned whatever the data's units
            extent = float((hi - lo).max())
            self.shared_program["u_eps"] = 1e-2 * extent if extent > 0 else 1e-2
        elif data["moved"]:
            # an empty visual draws nothing (_prepare_draw) and has no bounds
            # to contribute to a camera's set_range()
            self._bounds = None

        # new centers change the sort key and the slot count; a new
        # splats-per-row changes what the uploaded slots mean. Either way every
        # view's order is stale and must be rebuilt (see _sort).
        if data["moved"] or data["per_row_changed"]:
            self._sort_epoch += 1

        if data["degree_changed"]:
            self._sh_degree = data["degree"]
            self._sh_func = Function(_sh_shader(self._sh_degree))
            self._sh_func["sh_tex"] = self._sh_tex
            self.shared_program.vert["eval_sh"] = self._sh_func

        if geom is not None or color is not None:
            self._checked_texture_size = False
        if geom is not None:
            self._geom_tex.set_data(geom)
            self.shared_program["u_geom_size"] = (
                float(geom.shape[1]), float(geom.shape[0])
            )
        if color is not None:
            self._sh_tex.set_data(color)
            self._sh_func["sh_tex_size"] = (
                float(color.shape[1]), float(color.shape[0])
            )

    @staticmethod
    def _pack_geometry(data):
        """Build the geometry texture (centers, covariance, alpha)."""
        cov_a, cov_b = data["cov_a"], data["cov_b"]
        tex, records = _texture_for(
            len(data["pos"]), _GEOM_TEXELS, data["per_row"])
        records[:, 0, 0:3] = data["pos"]
        records[:, 0, 3] = data["alpha"]
        records[:, 1, 0] = cov_a[:, 0]                 # S00
        records[:, 1, 1] = cov_a[:, 1]                 # S01
        records[:, 1, 2] = cov_a[:, 2]                 # S02
        records[:, 1, 3] = cov_b[:, 0]                 # S11
        records[:, 2, 0] = cov_b[:, 1]                 # S12
        records[:, 2, 1] = cov_b[:, 2]                 # S22
        return tex

    @staticmethod
    def _pack_sh(data):
        """Build the color texture at the degree about to be rendered."""
        k = (data["degree"] + 1) ** 2
        tex, records = _texture_for(len(data["sh"]), k, data["per_row"])
        records[:, :, 0:3] = data["sh"][:, :k, :]
        return tex

    def _depth_gradient(self, view):
        """Gradient of framebuffer depth w.r.t. position (the view axis).

        The visual->framebuffer chain is affine, so framebuffer z is a linear
        function of position; four probes recover its gradient. ``pos @ grad``
        is then a back-to-front sort key that needs no per-point projective map
        or perspective divide, and its direction is the view axis used to
        decide when a re-sort is actually needed.
        """
        tr = view.get_transform("visual", "framebuffer")
        z = tr.map(_DEPTH_PROBES)[:, 2]
        return (z[1:] - z[0]).astype(np.float32)  # d(depth)/d(x, y, z)

    def _sort(self, view):
        """Re-sort and re-upload ``view``'s draw order if its direction moved."""
        grad = self._depth_gradient(view)
        norm = np.linalg.norm(grad)
        if norm == 0:
            # a transform that drops z leaves no depth to sort by, but the
            # order still has to name every splat or only the ones it happens
            # to hold get drawn
            if view._last_sort_epoch == self._sort_epoch:
                return
            # a zero direction never matches a real one, so the next
            # non-degenerate draw re-sorts
            view._last_view_dir = np.zeros(3, np.float32)
            view._last_sort_epoch = self._sort_epoch
            self._upload_order(view, np.arange(len(self._splat_pos)))
            return
        view_dir = grad / norm
        if (view._last_sort_epoch == self._sort_epoch
                and np.allclose(view_dir, view._last_view_dir)):
            return
        view._last_view_dir = view_dir
        view._last_sort_epoch = self._sort_epoch

        depth = self._splat_pos @ grad
        lo, hi = float(depth.min()), float(depth.max())
        if hi > lo:
            # rescale far -> near onto 0..65535 in place (depth is ours), then
            # quantize so the stable argsort is an O(N) radix sort
            depth -= hi
            depth *= -65535.0 / (hi - lo)
            order = np.argsort(depth.astype(np.uint16), kind="stable")
        else:
            # all splats share a depth plane: any order composites the same
            order = np.arange(len(depth))

        self._upload_order(view, order)

    def _upload_order(self, view, order):
        """Upload a draw order to ``view`` as (column, row) texture slots.

        Both components stay small enough to be exact in float32 however many
        splats there are, which a running index would not.
        """
        per_row = self._per_row
        slots = np.empty((len(order), 2), np.float32)
        slots[:, 0] = order % per_row      # column, in splats
        slots[:, 1] = order // per_row     # row
        view._slot_vbo.set_data(slots)

    def _check_texture_size(self, view):
        """Warn once if a data texture exceeds this context's real limit.

        _MAX_TEX_SIZE is a conservative guess made before a GL context exists;
        the actual GL_MAX_TEXTURE_SIZE is only knowable at draw time, and a
        texture over it fails to allocate with no useful message.
        """
        self._checked_texture_size = True
        canvas = getattr(view.transforms, "canvas", None)
        context = getattr(canvas, "context", None)
        if context is None:
            return
        limit = context.capabilities.get("max_texture_size")
        height, width = (max(t.shape[i] for t in (self._geom_tex, self._sh_tex))
                         for i in (0, 1))
        if limit and max(height, width) > limit:
            logger.warning(
                "GaussianSplatVisual needs a %d x %d data texture but this "
                "context supports at most %d in either dimension; the splats "
                "will not render. Use fewer splats or a lower sh_degree.",
                width, height, limit,
            )

    def _prepare_draw(self, view):
        if self._splat_pos is None or len(self._splat_pos) == 0:
            return False
        if not self._checked_texture_size:
            self._check_texture_size(view)
        self._sort(view)
        return True

    def _prepare_transforms(self, view):
        prog = view.view_program
        prog.vert["visual_to_framebuffer"] = view.get_transform("visual", "framebuffer")
        prog.vert["framebuffer_to_render"] = view.get_transform("framebuffer", "render")
        # for the per-splat view direction used by the SH color evaluation
        prog.vert["visual_to_document"] = view.get_transform("visual", "document")
        prog.vert["document_to_visual"] = view.get_transform("document", "visual")

    def _compute_bounds(self, axis, view):
        if self._bounds is None:
            return None
        return self._bounds[0, axis], self._bounds[1, axis]

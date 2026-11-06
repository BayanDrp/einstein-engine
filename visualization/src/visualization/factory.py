from manim import BLUE, GREEN, ORANGE, ORIGIN, PINK, TEAL, Dot3D, Sphere, VGroup, YELLOW

PALETTE = [
    BLUE,
    YELLOW,
    TEAL,
    ORANGE,
    PINK,
    GREEN,
]


def _particle(track, index):
    color = track.color or PALETTE[index % len(PALETTE)]
    radius = track.display_size or 0.10

    return VGroup(
        # halo first: drawn behind the core, same color, much fainter
        Dot3D(point=ORIGIN, radius=radius * 2.5, color=color, fill_opacity=0.2, resolution=(16, 16)),
        Dot3D(point=ORIGIN, radius=radius, color=color, fill_opacity=1.0, resolution=(16, 16)),
    )


def _matter(track, index):
    color = track.color or PALETTE[index % len(PALETTE)]
    radius = track.display_size or 0.30

    return VGroup(
        # glow shell first, then the readable core; fine mesh so the
        # big body reads as a smooth surface, not a low-poly ball
        Sphere(center=ORIGIN, radius=radius * 1.7, color=color, fill_opacity=0.12, resolution=(32, 16)),
        Sphere(center=ORIGIN, radius=radius, color=color, fill_opacity=0.85, resolution=(32, 16)),
    )

BUILDERS = {"particle": _particle, "matter": _matter}

def track_mobject(track, index):
    builder = BUILDERS.get(track.kind.value, _particle)
    return builder(track, index)

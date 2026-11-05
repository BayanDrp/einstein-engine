from manim import BLUE, GREEN, ORANGE, ORIGIN, PINK, TEAL, Dot3D, Sphere, YELLOW

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

    return Dot3D(
        point=ORIGIN,
        radius=radius,
        color=color,
    )


def _matter(track, index):
    color = track.color or PALETTE[index % len(PALETTE)]
    radius = track.display_size or 0.30

    return Sphere(
        center=ORIGIN,
        radius=radius,
        color=color,
        fill_opacity=0.7,
    )

BUILDERS = {"particle": _particle, "matter": _matter}

def track_mobject(track, index):
    builder = BUILDERS.get(track.kind.value, _particle)
    return builder(track, index)

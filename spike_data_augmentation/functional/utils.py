import warnings


def guess_event_ordering_numpy(events):
    warnings.warn("[SDAug]::Guessing the ordering of xytp in events")

    if np.issubdtype(events.dtype, np.numeric):
        guess = "xytp"
    else:
        raise NotImplementedError("Unable to guess event ordering")

    warnings.warn("[SDAug]::Guessed [%s] as ordering of events" % guess)

    return guess


def xytp_indices_from_ordering(ordering):
    x = ordering.index("x")
    y = ordering.index("y")
    t = ordering.index("t")
    p = ordering.index("p")

    return x, y, t, p


def ordering_from_xytp(x, y, t, p):
    ordering = "aaaa"

    ordering[x] = "x"
    ordering[y] = "y"
    ordering[t] = "t"
    ordering[p] = "p"

    if "a" in ordering:
        raise RuntimeError(
            "Event tuple dimensions not all accounted for [%s], need x,y,t, and p"
            % ordering
        )

    return ordering


def is_multi_image(images, sensor_size):
    warnings.warn("[SDAug]::Guessing if there are multiple images")
    guess = True
    warnings.warn("[SDAug]::Guessed [%s]" % str(guess))

    return guess

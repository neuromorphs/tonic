import numpy as np
import math


def to_bina_rep_numpy(
    event_frames: np.ndarray,
    n_frames: int = 1,
    n_bits: int = 8,
):
    """Representation that takes T*B binary event frames to produce a sequence of T frames of N-bit numbers.
    To do so, N binary frames are interpreted as a single frame of N-bit representation. Taken from the paper
    Barchid et al. 2022, Bina-Rep Event Frames: a Simple and Effective Representation for Event-based cameras
    https://arxiv.org/pdf/2202.13662.pdf

    Parameters:
        event_frames: numpy.ndarray of shape (T*BxPxHxW). The sequence of event frames.
        n_frames (int): the number T of bina-rep frames.
        n_bits (int): the number N of bits used in the N-bit representation.

    Returns:
        (numpy.ndarray) the sequence of bina-rep event frames with dimensions (TxPxHxW).
    """
    assert type(event_frames) == np.ndarray and len(event_frames.shape) == 4
    assert n_frames >= 1
    assert n_bits >= 2

    if event_frames.shape[0] != n_bits * n_frames:
        raise ValueError(
            "the input event_frames must have the right number of frames to the targeted"
            f"sequence of {n_frames} bina-rep event frames of {n_bits}-bit representation."
            f"Got: {event_frames.shape[0]} frames. Expected: {n_frames}x{n_bits}={n_bits * n_frames} frames."
        )

    event_frames = (event_frames > 0).astype(np.float32)  # get binary event_frames

    bina_rep_seq = np.zeros((n_frames, *event_frames.shape[1:]), dtype=np.float32)

    for i in range(n_frames):
        frames = event_frames[i * n_bits : (i + 1) * n_bits]
        bina_rep_frame = bina_rep(frames)
        bina_rep_seq[i] = bina_rep_frame

    return bina_rep_seq


def bina_rep(frames: np.ndarray) -> np.ndarray:
    """Computes one Bina-Rep frame from the sequence of N binary event-frames in parameter.

    Args:
        frames (numpy.ndarray): the sequence of N binary event frames used to compute the bina-rep frame. Shape=(NxPxHxW)

    Returns:
        numpy.ndarray: the resulting bina-rep event frame. Shape=(PxHxW)
    """
    mask = 2 ** np.arange(frames.shape[0] - 1, -1, -1, dtype=np.float32)
    arr_mask = [
        mask for _ in range(frames.shape[1] * frames.shape[2] * frames.shape[3])
    ]
    mask = np.stack(arr_mask, axis=-1)
    mask = np.reshape(mask, frames.shape)

    return np.sum(mask * frames, 0) / (2 ** mask.shape[0] - 1)

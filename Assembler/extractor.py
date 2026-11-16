import time
from typing import Any
import numpy as np

VALID_INTERLEAVES = (0, 1, 3, 5)


def build_interleave_mask(
    H: int, W: int, R: int, C: int, d: int
) -> np.ndarray:
    """
    Creating a 2D-boolean mask(H,W) for all interleave-pixels at given d. We create a empty 2D-mask (HxW) filled
    with False. We get the cell sizes (s_h, s_w) by integer division with R and C. We then check how many black
    (zeroes) pixels should be before (d // 2) and after (d - before). We then loop over the rows R and the columns C
    and put True on the exact rows and colums where the black frames should be.

    """
    if d not in VALID_INTERLEAVES:
        raise ValueError(f"not valid interleave d = {d}. Must belong to {VALID_INTERLEAVES}.")

    mask = np.zeros((H, W), dtype=bool)
    if d == 0:
        return mask

    s_h = H // R
    s_w = W // C
    before = d // 2  # q = floor(d / 2) pixels before the piece (top/left)
    after = d - before  # q + 1 pixels after the piece (bottom/right)

    for r in range(R):
        y0 = r * s_h
        y1 = (r + 1) * s_h
        if before > 0:
            mask[y0 : y0 + before, :] = True
        if after > 0:
            mask[y1 - after : y1, :] = True

    for c in range(C):
        x0 = c * s_w
        x1 = (c + 1) * s_w
        if before > 0:
            mask[:, x0 : x0 + before] = True
        if after > 0:
            mask[:, x1 - after : x1] = True

    return mask


def estimate_interleave(
    image: np.ndarray, R: int, C: int
) -> tuple[int, dict[str, Any]]:
    """
        Estimating interleave d in {0, 1, 3, 5}. Measuring execution time and controlling that 
        the image is an RGB-image with dimensions that are evenly divisible by R and C. We then
        test all the valid candidates in d. d = 0 is always a valid candidate. For 1, 3 and 5, we
        get the boolea-mask from build_interleave_mask and the pixels that are given as True
        are saved in masked_pixels = image[mask]. We check that all the pixels are zero in 
        is_exact_zero = np.all(masked_pixels == 0, axis = 1). If all the pixels are zero then
        the candidate is approved and appended to valid_candidates. Lastly we choose the max(valid_candidates).

        Parameters:
            image(np.ndarray): 8-bit RGB-image (H, W, 3).
            R(int): Rows.
            C(int): Columns.

        Returnerar:
            (d_hat, diagnostics)(tuple[int, dict[str, Any]]): d_hat is the estimated interleave value.
        """
    t0 = time.perf_counter()

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("The image must be a 3-channel RGB-image (H, W, 3).")

    H, W, _ = image.shape
    if H % R != 0 or W % C != 0:
        raise ValueError(
            f"The image dimensions ({H}x{W}) are not evenly divisible by R={R}, C={C}."
        )

    candidate_stats = {}
    valid_candidates = []

    for d in VALID_INTERLEAVES:
        if d == 0:
            candidate_stats[d] = {"zero_ratio": 1.0, "masked_pixels": 0}
            valid_candidates.append(d)
        continue

    mask = build_interleave_mask(H, W, R, C, d)
    masked_pixels = image[mask]  # Form: (N_mask, 3)

    # A pixel in the interleave is only valid if all three channels (R, G, B) == 0
    is_exact_zero = np.all(masked_pixels == 0, axis=1)
    zero_ratio = float(np.mean(is_exact_zero))

    candidate_stats[d] = {
        "zero_ratio": zero_ratio,
        "masked_pixels": int(masked_pixels.shape[0]),
    }

    if np.all(is_exact_zero):
        valid_candidates.append(d)

    # Choosing the greatest d value which 2D-mask contains only zeroes
    d_hat = max(valid_candidates)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    diagnostics = {
        "d_estimated": d_hat,
        "cell_height": H // R,
        "cell_width": W // C,
        "candidate_stats": candidate_stats,
        "interleave_estimation_ms": round(elapsed_ms, 3),
    }
    return d_hat, diagnostics


def extract_patches(
    image: np.ndarray, R: int, C: int, d: int | None = None
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """
    Estimating interleave (if d=None) and extracting R*C useable image pieces by rows. We calculate the image usable size
    where s_h and s_w are the total height and width of the cell, and p_h and p_w are the height and width of the cell 
    after we have removed theestimated d interleave. We cut out q = d_hat // 2 pixels from the left upper corner of each
    cell and cut out a sub matrix of size p_h x p_w to remove the interleave. We then save the bits in row order with index
    starting at 1 (k = (r_s)*C - c_s) and adding the bit size to diag.

    Indexkonvention:
        0-baserat index k_0 = r_s * C + c_s  (för r_s in 0..R-1, c_s in 0..C-1)
        1-baserat index k_1 = k_0 + 1        (motsvarar k = (r_s - 1)*C + c_s i specifikationen)
    """
    t0 = time.perf_counter()

    if d is None:
        d_hat, diag = estimate_interleave(image, R, C)
    else:
        if d not in VALID_INTERLEAVES:
            raise ValueError(f"Not valid interleave d={d}.")
        d_hat = d
        diag = {
            "d_estimated": d_hat,
            "cell_height": image.shape[0] // R,
            "cell_width": image.shape[1] // C,
            "interleave_estimation_ms": 0.0,
        }

    H, W, _ = image.shape
    s_h = H // R
    s_w = W // C
    p_h = s_h - d_hat
    p_w = s_w - d_hat

    if not (64 <= p_h <= 512 and 64 <= p_w <= 512):
        raise ValueError(
            f"Extracted image size ({p_h}x{p_w}) is outside of the allowed [64, 512] pixles."
        )

    q = d_hat // 2
    patches: list[np.ndarray] = []

    for r_s in range(R):
        for c_s in range(C):
            ys = r_s * s_h + q
            xs = c_s * s_w + q
            patch = image[ys : ys + p_h, xs : xs + p_w, :].copy()
            patches.append(patch)

    total_ms = (time.perf_counter() - t0) * 1000.0
    diag.update({
        "patch_height": p_h,
        "patch_width": p_w,
        "num_patches": len(patches),
        "extraction_total_ms": round(total_ms, 3),
    })
    return patches, diag
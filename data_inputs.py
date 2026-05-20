"""
Resolution of the per-input load list from a data config dict.

Two mutually exclusive input modes are supported:

(a) **Stacks mode** — multi-slice NIfTI volumes with a shared affine per stack.
    Use `subject.input_stacks` + `subject.input_masks`. The third axis of each
    volume is the slice axis; slice ID = z-voxel-index within the stack.

(b) **Slices mode** — a directory of motion-corrected single-slice NIfTI files
    (e.g. SVoRT output). Each file has its own world-space affine encoding the
    slice's already-corrected placement. Use `subject.slices_dir`.

Both modes feed the same downstream `preprocess_image` + assembly path in
`utils.load_data`: each "input" produces one or more slices, each with its own
affine, mask, and PSF covariance.
"""

import glob
import os
import re


def resolve_subject_inputs(cfg_data):
    """
    Build the per-input load list from a data config dict.

    Args:
        cfg_data: full data config dict (must contain a `subject` sub-dict).

    Returns:
        List of input-spec dicts. Each dict has:
            image_path:      str  — path to the NIfTI image (stack or slice).
            mask_path:       str | None — path to a brain mask, or None to
                                          fall back to non-zero voxels.
            slice_thickness: float — through-plane PSF FWHM in mm.
            label:           str  — human-readable identifier for logs.
    """
    subj = cfg_data['subject']
    has_stacks = bool(subj.get('input_stacks'))
    has_slices_dir = bool(subj.get('slices_dir'))

    if has_stacks == has_slices_dir:
        raise ValueError(
            "Specify exactly one of `subject.input_stacks` (stacks mode) or "
            "`subject.slices_dir` (slices mode), not both / neither."
        )

    st = subj.get('slice_thickness')
    if st is None:
        raise ValueError("`subject.slice_thickness` is required.")

    if has_stacks:
        return _resolve_stacks(subj, st)
    return _resolve_slices(subj, st)


def _resolve_stacks(subj, st):
    stacks = list(subj['input_stacks'])
    masks = list(subj.get('input_masks') or [])
    # Pad masks with None so each stack gets a matching entry
    masks += [None] * (len(stacks) - len(masks))
    masks = [m if m else None for m in masks]

    st_list = st if isinstance(st, list) else [st] * len(stacks)
    if len(st_list) != len(stacks):
        raise ValueError(
            f"`slice_thickness` has {len(st_list)} entries but "
            f"`input_stacks` has {len(stacks)} entries."
        )

    return [
        {'image_path': stacks[i],
         'mask_path': masks[i],
         'slice_thickness': st_list[i],
         'label': f"stack {i}"}
        for i in range(len(stacks))
    ]


def _resolve_slices(subj, st):
    """
    Discover slice + mask file pairs in `slices_dir`.

    Filename convention (configurable via `slice_glob` / `mask_prefix`):
        Slice files match `slice_glob` (default `[0-9]*.nii.gz`).
        For each slice, the paired mask is `{mask_prefix}{N}.nii.gz`
        where `N` is the trailing integer in the slice filename. Masks
        are optional — slices without a paired mask fall back to
        non-zero voxels.
    """
    slices_dir = subj['slices_dir']
    slice_glob = subj.get('slice_glob', '[0-9]*.nii.gz')
    mask_prefix = subj.get('mask_prefix', 'mask_')

    if not os.path.isdir(slices_dir):
        raise FileNotFoundError(f"`slices_dir` does not exist: {slices_dir}")

    candidates = glob.glob(os.path.join(slices_dir, slice_glob))
    if not candidates:
        raise FileNotFoundError(
            f"No slice files matched glob {slice_glob!r} in {slices_dir}"
        )
    candidates.sort(key=_filename_index)

    # slice_thickness: scalar / single-element list broadcasts; otherwise length must match
    if isinstance(st, list):
        if len(st) == 1:
            st_list = st * len(candidates)
        elif len(st) == len(candidates):
            st_list = st
        else:
            raise ValueError(
                f"`slice_thickness` has {len(st)} entries but {len(candidates)} "
                f"slices were discovered in {slices_dir}. Use a scalar to "
                f"broadcast, or supply exactly one entry per slice."
            )
    else:
        st_list = [st] * len(candidates)

    inputs = []
    for i, path in enumerate(candidates):
        idx = _filename_index(path)
        mask_candidate = os.path.join(slices_dir, f"{mask_prefix}{idx}.nii.gz")
        mask_path = mask_candidate if os.path.exists(mask_candidate) else None
        inputs.append({
            'image_path': path,
            'mask_path': mask_path,
            'slice_thickness': st_list[i],
            'label': f"slice {idx}",
        })

    n_with_mask = sum(1 for x in inputs if x['mask_path'] is not None)
    print(f"Slices mode: discovered {len(inputs)} slices in {slices_dir} "
          f"({n_with_mask} with paired masks).")
    return inputs


_TRAILING_INT_RE = re.compile(r'(\d+)(?=\.\w+(?:\.\w+)*$)')


def _filename_index(path):
    """Extract the trailing integer from a filename (used for natural sort)."""
    m = _TRAILING_INT_RE.search(os.path.basename(path))
    return int(m.group(1)) if m else 0

from __future__ import annotations

import logging
from typing import Optional

from heudiconv.utils import SeqInfo

lgr = logging.getLogger("heudiconv")


def create_key(
    template: Optional[str],
    outtype: tuple[str, ...] = ("nii.gz",),
    annotation_classes: None = None,
) -> tuple[str, tuple[str, ...], None]:
    if template is None or not template:
        raise ValueError("Template must be a valid format string")
    return (template, outtype, annotation_classes)


def infotodict(
    seqinfo: list[SeqInfo],
) -> dict[tuple[str, tuple[str, ...], None], list[str]]:
    """Heuristic evaluator for determining which runs belong where

    allowed template fields - follow python string module:

    item: index within category
    subject: participant id
    seqitem: run number during scanning
    subindex: sub index within group
    """

    ###### Definitions #######

    t1w = create_key('sub-{subject}/anat/sub-{subject}_T1w')

    # Save the RPE (reverse phase-encode) B0 image as a fieldmap (fmap).  It will be used to correct
    # the distortion in the DWI
    fmap_pa =  create_key('sub-{subject}/fmap/sub-{subject}_dir-PA_run-{item:03d}_epi')
    fmap_ap =  create_key('sub-{subject}/fmap/sub-{subject}_dir-AP_run-{item:03d}_epi')

    func_run = create_key('sub-{subject}/func/sub-{subject}_task-switch_run-{item:03d}_bold')
    func_sbref = create_key('sub-{subject}/func/sub-{subject}_task-switch_run-{item:03d}_sbref')
    
    info: dict[tuple[str, tuple[str, ...], None], list[str]] = {t1w: [], fmap_pa: [], fmap_ap: [], func_run: [], func_sbref: []}

    for idx, s in enumerate(seqinfo):

        if ('T1' in s.protocol_name):
            info[t1w].append(s.series_id)

        if ('PA' in s.protocol_name):
            info[fmap_pa].append(s.series_id)

        if ('AP' in s.protocol_name):
            info[fmap_ap].append(s.series_id)

		# dim3 == 60 is important to exclude the last incomplete volume (otherwise it
		# stores as a separate file, and hinders BIDS formatting)
        if ('switch' in s.protocol_name) and ('SBRef' not in s.series_description) and (s.dim3 == 60) and (s.is_derived == False):
            info[func_run].append(s.series_id)
            print(s.protocol_name)
            print(s.series_description)
            print(s.series_id)

        if ('SBRef' in s.series_description):
            info[func_sbref].append(s.series_id)
            print(s.protocol_name)
            print(s.series_description)
            print(s.series_id)

    print(info)

    return info

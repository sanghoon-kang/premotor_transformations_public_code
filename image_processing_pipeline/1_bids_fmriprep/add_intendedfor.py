"""Fill in IntendedFor, TotalReadoutTime and TaskName in the BIDS JSON sidecars."""
from __future__ import print_function
import json
import os
import os.path as op

import nibabel as nib
from bids import BIDSLayout


def intended_for_gen(fmap_nifti, niftis):
    """IntendedFor = the scans after this fieldmap, up to the next fieldmap of the same direction."""
    intended_for = []
    fmap_entities = fmap_nifti.get_entities()
    acq_time = fmap_nifti.get_metadata()['AcquisitionTime']
    out_dict = {}
    for nifti in niftis:
        nifti_meta = nifti.get_metadata()
        if nifti_meta['AcquisitionTime'] <= acq_time:
            continue
        if (nifti_meta['AcquisitionTime'] in out_dict) and \
                (nifti not in out_dict[nifti_meta['AcquisitionTime']]):
            out_dict[nifti_meta['AcquisitionTime']].append(nifti)
        elif nifti_meta['AcquisitionTime'] not in out_dict:
            out_dict[nifti_meta['AcquisitionTime']] = [nifti]
    for num in sorted([x for x in out_dict]):
        target_entities = [x.get_entities() for x in out_dict[num]]
        print(target_entities)
       
        # only stop at a fieldmap with the same direction
        if target_entities[0]['datatype'] == 'fmap':
            if target_entities[0]['direction'] == fmap_entities['direction']:
                thing_found = any([all([fmap_entities[x] == i[x] for x in fmap_entities if x != 'run'])
                                   for i in target_entities])
                if thing_found:
                    break
                else:
                    continue
        if 'session' in target_entities[0]:
            if target_entities[0]['datatype'] != 'fmap':
                intended_for.extend(sorted([op.join('ses-{0}'.format(target_entities[0]['session']),
                                                    target_entities[0]['datatype'],
                                                    x.filename) for x in out_dict[num]]))
        else:
            if target_entities[0]['datatype'] != 'fmap':
                intended_for.extend(sorted([op.join(target_entities[0]['datatype'],
                                                    x.filename) for x in out_dict[num]]))
    print(intended_for)
    return sorted(intended_for)


def complete_jsons(bids_dir, subs, ses, overwrite):
    """Update the JSON sidecars for the given subjects."""
    layout = BIDSLayout(op.abspath(bids_dir), validate=False)
    for sid in subs:
        print(sid)
        if ses:
            print("yes ses")
            niftis = layout.get(subject=sid, session=ses,
                                extension='nii.gz',
                                datatype=['func', 'fmap', 'anat'])
        else:
            print("no ses")
            niftis = layout.get(subject=sid,
                                extension='nii.gz',
                                datatype=['func', 'fmap', 'anat'])
            print(niftis)
        for nifti in niftis:
            
            data = nifti.get_metadata()
            print(data)
            print("going")
            dump = 0
            json_path = nifti.path.replace('.nii.gz', '.json')
            if ('EffectiveEchoSpacing' in data.keys()) and \
                    (overwrite or 'TotalReadoutTime' not in data.keys()):
                # adapted from fMRIPrep
                pe_idx = {'i': 0, 'j': 1, 'k': 2}[data['PhaseEncodingDirection'][0]]
                etl = nib.load(nifti.path).shape[pe_idx] \
                    // float(data.get('ParallelReductionFactorInPlane', 1.0))
                ees = data.get('EffectiveEchoSpacing', None)
                if ees is None:
                    raise Exception('Field "EffectiveEchoSpacing" not '
                                    'found in json')
                data['TotalReadoutTime'] = ees * (etl - 1)
                dump = 1
            if 'task' in nifti.get_entities() and (overwrite or 'TaskName' not in data.keys()):
                data['TaskName'] = nifti.get_entities()['task']
                dump = 1
            if (nifti.get_entities()['datatype'] == 'fmap') \
                    and (overwrite or 'IntendedFor' not in data.keys()):
                data['IntendedFor'] = intended_for_gen(nifti, niftis)
                dump = 1
            if dump == 1:
                os.chmod(json_path, 0o777)
                with open(json_path, 'w') as f_obj:
                    json.dump(data, f_obj, sort_keys=True, indent=4)

complete_jsons(bids_dir="/home/sk2949/project/mrdecode/bids_imgs/", subs=['mrd1011'
                                                                         ], ses=None, overwrite=True)
print("done")


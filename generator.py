import glob
import math
import os
import types
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import conv3d
import nibabel as nib
from scipy.io.matlab import loadmat
import csv
import re
# import ext.interpol as interpol
from pathology_primitives import ROSE_CRITERION_K
# Same constant used for both the label-level geometric gate
# (pathology_primitives.py's augment_batch) and the image-level
# visibility check below -- see its definition there for citation.


def _build_generator_context(datadir, label_list_segmentation, n_neutral_labels,
                              label_list_normalization, list_to_kill_photo,
                              list_to_kill_photo_inc_ventricles, siz, dtype, device,
                              healthy_only=False):
    """Everything sampled/built ONCE per generator instance (not per case):
    dataset listing, coordinate grid, lookup tables. Shared by super_generator
    and paired_pathology_generator."""
    # Initialize at the top of the scope to prevent UnboundLocalError
    datasets = []
    names = []
    
    # Collect list of available images, per dataset
    g = glob.glob(os.path.join(datadir, '*' + 'T1w.nii'))
    for i in range(len(g)):
        filename = os.path.basename(g[i])
        dataset = filename[:filename.find('.')]
        found = False
        for d in datasets:
            if dataset == d:
                found = True
        if found is False:
            datasets.append(dataset)
    print('Found ' + str(len(datasets)) + ' datasets with ' + str(len(g)) + ' scans in total')
    if len(datasets) == 0:
        raise FileNotFoundError(
            f"No *T1w.nii files found in datadir={datadir!r}. Check the path "
            f"exists and hasn't been renamed (`ls {datadir}`)."
        )

    if healthy_only:
        print('Applying HEALTHY ONLY filter to datasets...')
        
        def clean_id(s):
            # Strips non-alphanumeric chars and drops BIDS 'sub' prefix for robust matching
            return ''.join(c for c in s if c.isalnum()).replace('sub', '')
            
        def _oasis3_healthy_ids(participants_csv, cdr_csv, max_days=365):
            """sub-XXXX ids whose nearest CDR visit (within max_days of the
            scan day encoded in MR_ID, e.g. OAS30001_MR_d0129) is CDR 0."""
            h_ids = set()
            if not os.path.isfile(cdr_csv):
                print(f"  [Warning] OASIS3 CDR table not found at {cdr_csv}; OASIS3 excluded. "
                      f"Expected columns: OASISID, days_to_visit, CDRTOT (OASIS3 ADRC clinical data).")
                return h_ids
            cdr = {}
            with open(cdr_csv, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                cols = {c.lower(): c for c in (reader.fieldnames or [])}
                c_id = next((cols[k] for k in ('oasisid', 'subject', 'oasis_id') if k in cols), None)
                c_day = next((cols[k] for k in ('days_to_visit', 'days', 'day') if k in cols), None)
                c_cdr = next((cols[k] for k in ('cdrtot', 'cdr', 'cdr_total') if k in cols), None)
                if not (c_id and c_day and c_cdr):
                    print(f"  [Warning] OASIS3 CDR table {cdr_csv}: need id/days/cdr columns, have {reader.fieldnames}")
                    return h_ids
                for row in reader:
                    try:
                        cdr.setdefault(row[c_id].strip(), []).append((int(float(row[c_day])), float(row[c_cdr])))
                    except (ValueError, KeyError):
                        continue
            n_seen = 0
            with open(participants_csv, 'r', encoding='utf-8-sig') as f:
                for row in csv.DictReader(f):
                    mr, sub = row.get('MR_ID', ''), row.get('subj_ID', '')
                    m = re.match(r'(OAS3\d+)_MR_d(\d+)', mr)
                    if not m or not sub:
                        continue
                    n_seen += 1
                    visits = cdr.get(m.group(1), [])
                    if not visits:
                        continue
                    day = int(m.group(2))
                    d, score = min(visits, key=lambda v: abs(v[0] - day))
                    if abs(d - day) <= max_days and score == 0.0:
                        h_ids.add(clean_id(sub))
            print(f"  OASIS3: {len(h_ids)} of {n_seen} scans have CDR 0 within {max_days} days.")
            return h_ids

        def _get_healthy_ids(filepath):
            h_ids = set()
            if not os.path.isfile(filepath):
                print(f"  [Warning] {filepath} not found!")
                return h_ids
            
            # Common names for Subject ID and Diagnosis columns
            id_cols = ['participant_id', 'subject', 'subject_id', 'id']
            dx_cols = ['group', 'research_group', 'research group', 'diagnosis', 'dx1', 'dx', 'cdr']
            
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                # Auto-detect delimiter
                first_line = f.readline()
                delim = '\t' if '\t' in first_line else ','
                f.seek(0)
                
                reader = csv.DictReader(f, delimiter=delim)
                headers_lower = [h.lower().strip() for h in (reader.fieldnames or [])]
                
                actual_id_col = next((orig for orig, lower in zip(reader.fieldnames, headers_lower) if lower in id_cols), None)
                actual_dx_col = next((orig for orig, lower in zip(reader.fieldnames, headers_lower) if lower in dx_cols), None)
                
                if not actual_id_col or not actual_dx_col:
                    print(f"  [Warning] Missing ID or DX columns in {filepath}.")
                    print(f"  Available headers: {reader.fieldnames}")
                    return h_ids
                    
                for row in reader:
                    dx_val = str(row[actual_dx_col]).lower().strip()
                    # Indicators for control/healthy/cognitively normal
                    if dx_val in ['cn', 'control', 'normal', 'hc', '0', '0.0', 'healthy'] or 'cognitively normal' in dx_val:
                        h_ids.add(clean_id(str(row[actual_id_col])))
            return h_ids

        # OASIS3's participants.csv carries no diagnosis (and neither does
        # any other table stored with the images), so healthy status has to
        # come from the ADRC clinical CDR table of the OASIS3 release. Drop
        # it at OASIS3_CDR_CSV; until then OASIS3 is excluded, loudly.
        oasis_ids = _oasis3_healthy_ids(os.path.join(os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_datasets'), 'OASIS3/participants.csv'),
                                        OASIS3_CDR_CSV) if 'OASIS3' in HEALTHY_EXTRA_DATASETS else set()
        # 2026-10-07: the healthy-cohort split. The sessions listed in OASIS3_EXCLUDE_FILE (first token per line = the curated
        # sub-NNNN id; the file is the evaluation cohort) are NEVER admitted to the pool, so the cohort stays held out of every
        # model trained with --oasis. Logged loudly so a run without the exclusion is visible in its log.
        if oasis_ids and OASIS3_EXCLUDE_FILE:
            if os.path.isfile(OASIS3_EXCLUDE_FILE):
                _excl = {ln.split()[0] for ln in open(OASIS3_EXCLUDE_FILE) if ln.strip() and not ln.startswith('#')}
                _n0 = len(oasis_ids); oasis_ids = {i for i in oasis_ids if i not in _excl}
                print(f"  OASIS3: {_n0} CDR-0 sessions, {_n0 - len(oasis_ids)} excluded as the held-out healthy cohort ({OASIS3_EXCLUDE_FILE}), {len(oasis_ids)} admitted")
            else:
                raise SystemExit(f"--oasis_exclude: file not found: {OASIS3_EXCLUDE_FILE}")
        elif oasis_ids:
            print(f"  [Warning] OASIS3 admitted WITHOUT a held-out exclusion ({len(oasis_ids)} CDR-0 sessions): this model cannot be evaluated on the OASIS3 healthy cohort")
        adni_ids = _get_healthy_ids(os.path.join(os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_datasets'), 'ADNI3/participants.tsv')) \
            if 'ADNI3' in HEALTHY_EXTRA_DATASETS else set()
        
        # Buckner40 is OPT-IN (train.py --buckner40), off by default, 2026-10-03. It spans young, old and
        # DEMENTED adults (Buckner et al. 2004), so by default it violates the no-diagnosis rule for the
        # anatomy prior. It used to sit in this list with the correction living one file away in the
        # launcher's --exclude_datasets, which made THIS LINE look like the pool definition; it was copied
        # as such twice in one day. The default must be right on its own. An explicit --exclude_datasets
        # still wins over the opt-in, so the launchers' belt-and-braces flag keeps working.
        _base = (['Buckner40'] if ALLOW_BUCKNER40 else []) + ['Chinese-HCP', 'HCP']
        healthy_datasets_all = [d for d in _base if d not in EXCLUDE_DATASETS]
        if ALLOW_BUCKNER40 and 'Buckner40' in healthy_datasets_all:
            print("  [Warning] --buckner40: admitting Buckner40 UNFILTERED. Unlike ADNI3/OASIS3 there is no "
                  "diagnosis table for it, and the cohort contains demented subjects (Buckner et al. 2004).")
        if EXCLUDE_DATASETS:
            print(f"  Excluded by --exclude_datasets: {', '.join(EXCLUDE_DATASETS)}")
        # The elderly cohorts are opt-in (train.py --adni, --oasis) so existing
        # baselines, which were trained on the three young cohorts only,
        # keep their data pool.
        healthy_datasets_partial = {d: ids for d, ids in (('OASIS3', oasis_ids), ('ADNI3', adni_ids))
                                    if d in HEALTHY_EXTRA_DATASETS}
        print(f"  Elderly cohorts (--adni / --oasis): {'on: ' + ', '.join(HEALTHY_EXTRA_DATASETS) if HEALTHY_EXTRA_DATASETS else 'off'}")
        
        active_datasets = []
        total_healthy_scans = 0
        
        for d in datasets:
            if d not in healthy_datasets_all and d not in healthy_datasets_partial:
                continue
                
            dataset_files = glob.glob(os.path.join(datadir, d + '.*' + 'T1w.nii'))
            
            if d in healthy_datasets_partial:
                valid_ids = healthy_datasets_partial[d]
                # Training files are <dataset>.<sub-id>.T1w.nii; match the
                # BIDS id exactly. The previous substring match required
                # len(id) > 3, which silently dropped every 3-digit ADNI3 id.
                filtered = [f for f in dataset_files
                            if clean_id(os.path.basename(f).split('.')[1]) in valid_ids]
                dataset_files = filtered
                print(f"  Filtered {d}: kept {len(dataset_files)} healthy scans "
                      f"(of {len(glob.glob(os.path.join(datadir, d + '.*T1w.nii')))}; {len(valid_ids)} healthy ids).")
            else:
                print(f"  Included {d}: {len(dataset_files)} scans (All assumed healthy).")
                
            if len(dataset_files) > 0:
                names.append(dataset_files)
                active_datasets.append(d)
                total_healthy_scans += len(dataset_files)
                
        datasets = active_datasets
        print(f"HEALTHY ONLY filter complete. Kept {len(datasets)} datasets with {total_healthy_scans} scans in total.")
    else:
        for i in range(len(datasets)):
            names.append(glob.glob(os.path.join(datadir, datasets[i] + '.*' + 'T1w.nii')))

    print("names lengths =", [len(x) for x in names])

    if len(names) == 0 or all(len(x) == 0 for x in names):
        raise RuntimeError(
            "The 'names' list is empty! If --healthy is on, the CSV parsing may have filtered out everyone. "
            "Check the console warnings above to see if the ID/Diagnosis columns were found."
        )
    contrasts = _contrast_inventory(datasets, names) if IMAGE_SOURCE == 'real' else None   # data-kind arms (2026-10-20)

    # Get resolution and maximum modeled distance for white / pial surfaces
    aux = nib.load(names[0][0][:-7] + 'lp_dist_map.nii')
    res_training_data = np.sum(aux.affine ** 2, axis=0)[:-1]
    max_surf_dist_modeled = np.max(0.05 * (aux.get_fdata() - 128))

    with torch.no_grad():
        # prepare grid
        print('Preparing grid...')
        xx, yy, zz = np.meshgrid(range(siz[0]), range(siz[1]), range(siz[2]), sparse=False, indexing='ij')
        xx = torch.tensor(xx, dtype=dtype, device=device)
        yy = torch.tensor(yy, dtype=dtype, device=device)
        zz = torch.tensor(zz, dtype=dtype, device=device)
        c = torch.tensor((np.array(siz) - 1) / 2, dtype=dtype, device=device)
        xc = xx - c[0]
        yc = yy - c[1]
        zc = zz - c[2]

        # Matrix for one-hot encoding (includes a lookup-table)
        n_labels = len(label_list_segmentation)
        lut = torch.zeros(10000, dtype=torch.long, device=device)
        for l in range(n_labels):
            lut[label_list_segmentation[l]] = l
        onehotmatrix = torch.eye(n_labels, dtype=dtype, device=device)

        # Array to quickly extract normalization mask from segmentation
        lut_normalization = torch.zeros(10000, dtype=torch.bool, device=device)
        for lln in label_list_normalization:
            lut_normalization[lln] = True

        # Array to kill background labels in photo mode if needed
        lut_kill_photo = torch.arange(0, 10000, dtype=torch.int32, device=device)
        for lln in list_to_kill_photo:
            lut_kill_photo[lln] = 0
        lut_kill_photo_inc_ventricles = torch.arange(0, 10000, dtype=torch.int32, device=device)
        for lln in list_to_kill_photo_inc_ventricles:
            lut_kill_photo_inc_ventricles[lln] = 0

        # Vector to flip left and right. The label list is [neutral | left block | right block]; labels appended after
        # that layout (CHOROID_TARGET: 57 left, 58 right) are swapped pairwise so the existing compact ids never move.
        n_base = n_labels - sum(1 for r in CHOROID_RAW if r in label_list_segmentation)
        nlat = int((n_base - n_neutral_labels) / 2.0)
        vflip = np.concatenate([np.array(range(n_neutral_labels)),
                                np.array(range(n_neutral_labels + nlat, n_base)),
                                np.array(range(n_neutral_labels, n_neutral_labels + nlat))])
        if n_base < n_labels:
            l, r = label_list_segmentation.index(CHOROID_RAW[0]), label_list_segmentation.index(CHOROID_RAW[1])
            extra = np.arange(n_base, n_labels); extra[l - n_base], extra[r - n_base] = r, l
            vflip = np.concatenate([vflip, extra])

        print('Generator is ready!')

    return types.SimpleNamespace(
        datasets=datasets, names=names, res_training_data=res_training_data,
        max_surf_dist_modeled=max_surf_dist_modeled, xx=xx, yy=yy, zz=zz, c=c,
        xc=xc, yc=yc, zc=zc, n_labels=n_labels, lut=lut, onehotmatrix=onehotmatrix,
        lut_normalization=lut_normalization, lut_kill_photo=lut_kill_photo,
        lut_kill_photo_inc_ventricles=lut_kill_photo_inc_ventricles, vflip=vflip, contrasts=contrasts,
    )


def _load_and_prepare_case(ctx, mode_sampler, sides, labels_to_kill,
                            max_rotation, max_shear, max_scaling,
                            nonlin_scale_min, nonlin_scale_max, nonlin_std_max,
                            bag_scale_min, bag_scale_max, bag_prob,
                            siz, dtype, device,
                            random_kill_labels=None, random_kill_prob=0.0,
                            random_kill_frac=(0.5, 1.0), minimal_io=True):
    """Selects a random case, loads it, samples the affine+nonlinear
    deformation and crop margins, and applies label-killing/normalization/
    exvivo/photo-mode handling -- everything before GMM intensity synthesis.
    Shared by super_generator and paired_pathology_generator (the latter
    injects pathology into G/S right after this returns, before any shared
    downstream randomness is consumed)."""
    # Select random case
    d_idx = np.random.randint(len(ctx.datasets))
    idx = np.random.randint(len(ctx.names[d_idx]))
    t1 = ctx.names[d_idx][idx]
    t1dm = ctx.names[d_idx][idx][:-7] + 'T1w.defacingmask.nii'
    t2 = ctx.names[d_idx][idx][:-7] + 'T2w.nii'
    t2dm = ctx.names[d_idx][idx][:-7] + 'T2w.defacingmask.nii'
    flair = ctx.names[d_idx][idx][:-7] + 'FLAIR.nii'
    flairdm = ctx.names[d_idx][idx][:-7] + 'FLAIR.defacingmask.nii'
    ct = ctx.names[d_idx][idx][:-7] + 'CT.nii'
    ctdm = ctx.names[d_idx][idx][:-7] + 'CT.defacingmask.nii'
    generation_labels = ctx.names[d_idx][idx][:-7] + 'generation_labels.nii'
    brain_dist_map = ctx.names[d_idx][idx][:-7] + 'brain_dist_map.nii'
    lp_dist_map = ctx.names[d_idx][idx][:-7] + 'lp_dist_map.nii'
    rp_dist_map = ctx.names[d_idx][idx][:-7] + 'rp_dist_map.nii'
    lw_dist_map = ctx.names[d_idx][idx][:-7] + 'lw_dist_map.nii'
    rw_dist_map = ctx.names[d_idx][idx][:-7] + 'rw_dist_map.nii'
    mni_reg_x = ctx.names[d_idx][idx][:-7] + 'mni_reg.x.nii'
    mni_reg_y = ctx.names[d_idx][idx][:-7] + 'mni_reg.y.nii'
    mni_reg_z = ctx.names[d_idx][idx][:-7] + 'mni_reg.z.nii'
    if sides == 'both':
        segmentation_labels = ctx.names[d_idx][idx][:-7] + 'brainseg_with_extracerebral.nii'
    elif sides == 'hemis':
        segmentation_labels = ctx.names[d_idx][idx][:-7] + 'brainseg.nii'
    else:
        raise Exception('sides must be both or hemi')

    t1dm = t1dm if os.path.isfile(t1dm) else None
    t2 = t2 if os.path.isfile(t2) else None
    t2dm = t2dm if os.path.isfile(t2dm) else None
    flair = flair if os.path.isfile(flair) else None
    flairdm = flairdm if os.path.isfile(flairdm) else None
    ct = ct if os.path.isfile(ct) else None
    ctdm = ctdm if os.path.isfile(ctdm) else None
    rp_dist_map = rp_dist_map if os.path.isfile(rp_dist_map) else None
    rw_dist_map = rw_dist_map if os.path.isfile(rw_dist_map) else None

    # Load generation labels off the bat
    Gimg = nib.load(generation_labels)
    mode, resolution, thickness = mode_sampler()

    # sample affine deformation
    rotations = (2 * max_rotation * np.random.rand(3) - max_rotation) / 180.0 * np.pi
    shears = (2 * max_shear * np.random.rand(3) - max_shear)
    scalings = 1 + (2 * max_scaling * np.random.rand(3) - max_scaling)
    scaling_factor_distances = np.prod(scalings) ** .33333333333  # we divide distance maps by this, not perfect, but better than nothing
    A = torch.tensor(make_affine_matrix(rotations, shears, scalings), dtype=torch.float, device=device)

    # sample center
    max_shift = (torch.tensor(np.array(Gimg.shape[0:3]) - siz, dtype=dtype, device=device)) / 2
    max_shift[max_shift < 0] = 0
    c2 = torch.tensor((np.array(Gimg.shape[0:3]) - 1) / 2, dtype=dtype, device=device) + (2 * (max_shift * torch.rand(3, dtype=float, device=device)) - max_shift)

    # sample nonlinear deformation (photos a bit special)
    nonlin_scale = nonlin_scale_min + np.random.rand(1) * (nonlin_scale_max - nonlin_scale_min)
    siz_F_small = np.round(nonlin_scale * np.array(siz)).astype(int).tolist()
    if mode == 'photo':
        siz_F_small[1] = np.round(siz[1] / resolution[1]).astype(int)
    nonlin_std = nonlin_std_max * np.random.rand()
    Fsmall = nonlin_std * torch.randn([*siz_F_small, 3], dtype=dtype, device=device)
    Field = myzoom_torch(Fsmall, np.array(siz) / siz_F_small, device)
    if mode == 'photo':
        Field[:, :, :, 1] = 0

    # deformed coordinates (we do nonlinear "first" ie after so we can do heavy coronal deformations in photo mode)
    xx1 = ctx.xc + Field[:, :, :, 0]
    yy1 = ctx.yc + Field[:, :, :, 1]
    zz1 = ctx.zc + Field[:, :, :, 2]
    xx2 = A[0, 0] * xx1 + A[0, 1] * yy1 + A[0, 2] * zz1 + c2[0]
    yy2 = A[1, 0] * xx1 + A[1, 1] * yy1 + A[1, 2] * zz1 + c2[1]
    zz2 = A[2, 0] * xx1 + A[2, 1] * yy1 + A[2, 2] * zz1 + c2[2]

    # Get the margins for reading images
    x1 = max(0, torch.floor(torch.min(xx2)).int().cpu().numpy())
    y1 = max(0, torch.floor(torch.min(yy2)).int().cpu().numpy())
    z1 = max(0, torch.floor(torch.min(zz2)).int().cpu().numpy())
    x2 = min(Gimg.shape[0], 1 + torch.ceil(torch.max(xx2)).int().cpu().numpy())
    y2 = min(Gimg.shape[1], 1 + torch.ceil(torch.max(yy2)).int().cpu().numpy())
    z2 = min(Gimg.shape[2], 1 + torch.ceil(torch.max(zz2)).int().cpu().numpy())
    xx2 -= int(x1)
    yy2 -= int(y1)
    zz2 -= int(z1)

    # Read in data
    # Read only the bounding box, not the whole volume. get_fdata() decodes
    # every voxel and then discards everything outside [x1:x2, y1:y2, z1:z2];
    # nibabel's dataobj slices lazily and reads just that block off disk,
    # roughly 5x faster on a typical volume/bbox size. The generator is
    # I/O-bound on shared storage, so this dominates fresh-draw cost far
    # more than any GPU-side optimization would.
    G = torch.squeeze(torch.tensor(np.asarray(Gimg.dataobj[x1:x2, y1:y2, z1:z2]), dtype=torch.int, device=device))
    S = torch.squeeze(torch.tensor(np.asarray(nib.load(segmentation_labels).dataobj[x1:x2, y1:y2, z1:z2]), dtype=torch.int, device=device))
    if CHOROID_TARGET:
        S = inject_choroid_target(S, G)
    T1 = torch.squeeze(torch.tensor(np.asarray(nib.load(t1).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device))
    # minimal_io: with mix_real_images=False and mode='invivo', most of these
    # volumes are loaded, cropped, deformed and returned without ever being
    # read. T2/FLAIR/CT fed only the linear-combination branch that
    # mix_real_images disables; brain_dist_map is exvivo-only; the lp/rp/lw/rw
    # maps are used only in the exvivo_only_cerebrum bag; the MNI maps and
    # defacing masks are returned but never consumed by the training loop.
    # That is 15 of 18 loads off a network filesystem, plus a deformation
    # each. Set minimal_io=False to restore them if a downstream consumer
    # needs them.
    _skip = minimal_io
    _skip_c = _skip and IMAGE_SOURCE != 'real'   # real-image arms draw among the subject's contrasts (2026-10-20)
    T1dm = None if (t1dm is None or _skip) else torch.squeeze(torch.tensor(np.asarray(nib.load(t1dm).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device))
    T2 = None if (t2 is None or _skip_c) else torch.squeeze(torch.tensor(np.asarray(nib.load(t2).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device))
    T2dm = None if (t2dm is None or _skip) else torch.squeeze(torch.tensor(np.asarray(nib.load(t2dm).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device))
    FLAIR = None if (flair is None or _skip_c) else torch.squeeze(torch.tensor(np.asarray(nib.load(flair).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device))
    FLAIRdm = None if (flairdm is None or _skip) else torch.squeeze(torch.tensor(np.asarray(nib.load(flairdm).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device))
    CT = None if (ct is None or _skip) else torch.squeeze(torch.tensor(np.asarray(nib.load(ct).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device))
    CTdm = None if (ctdm is None or _skip) else torch.squeeze(torch.tensor(np.asarray(nib.load(ctdm).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device))
    D = 0.1 * torch.squeeze(torch.tensor(np.asarray(nib.load(brain_dist_map).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device))
    LP = None if _skip else 0.05 * (torch.squeeze(torch.tensor(np.asarray(nib.load(lp_dist_map).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device)) - 128.0)
    RP = None if (rp_dist_map is None or _skip) else (0.05 * (torch.squeeze(torch.tensor(np.asarray(nib.load(rp_dist_map).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device)) - 128.0))
    LW = None if _skip else 0.05 * (torch.squeeze(torch.tensor(np.asarray(nib.load(lw_dist_map).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device)) - 128.0)
    RW = None if (rw_dist_map is None or _skip) else (0.05 * (torch.squeeze(torch.tensor(np.asarray(nib.load(rw_dist_map).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device)) - 128.0))
    MNIx = None if _skip else 0.0001 * torch.squeeze(torch.tensor(np.asarray(nib.load(mni_reg_x).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device))
    MNIy = None if _skip else 0.0001 * torch.squeeze(torch.tensor(np.asarray(nib.load(mni_reg_y).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device))
    MNIz = None if _skip else 0.0001 * torch.squeeze(torch.tensor(np.asarray(nib.load(mni_reg_z).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device))

    if IMAGE_SOURCE == 'real':   # a companion crop off the T1 box is not voxel-aligned: dropped, as _kseg_volumes does
        T2 = None if (T2 is not None and T2.shape != T1.shape) else T2
        FLAIR = None if (FLAIR is not None and FLAIR.shape != T1.shape) else FLAIR

    # kill labels if needed
    if labels_to_kill is not None:
        for ltk in labels_to_kill:
            S[S == ltk] = 0

    # normalize targets
    Mnorm = ctx.lut_normalization[S]
    if torch.sum(Mnorm) > 100:
        T1 /= torch.median(T1[Mnorm])
        T2 = None if T2 is None else (T2 / torch.median(T2[Mnorm]))
        FLAIR = None if FLAIR is None else (FLAIR / torch.median(FLAIR[Mnorm]))
        CT = None if CT is None else CT / 1000
    else:  # should not happen too often...
        T1 /= (1e-6 + torch.max(T1))
        T2 = None if T2 is None else (T2 / (1e-6 + torch.max(T2)))
        FLAIR = None if FLAIR is None else (FLAIR / (1e-6 + torch.max(FLAIR)))
        CT = None if CT is None else CT / 1000

    # Randomly remove a subset of structures entirely.
    #
    # Placed AFTER normalization on purpose: normalization takes the median
    # of T1 over label_list_normalization (cerebral WM), so killing those
    # first would divide by the median of an all-zero region.
    #
    # Unlike `labels_to_kill`, which only sets S to 0 (leaving the tissue
    # still synthesized and visible, just labelled background), this zeroes
    # G, S, and every intensity channel together -- so the tissue is absent
    # from the image AND from the label map, which is what real removal
    # (e.g. skull stripping) actually looks like.
    #
    # Label ids are in the 0-56 "brainseg_with_extracerebral" scheme that S
    # is loaded in, so e.g. range(34, 47) is the extracerebral block.
    if random_kill_labels and np.random.rand() < random_kill_prob:
        candidates = list(random_kill_labels)
        lo, hi = random_kill_frac
        frac = lo + np.random.rand() * (hi - lo)
        n_kill = int(round(frac * len(candidates)))
        if n_kill > 0:
            chosen = np.random.choice(candidates, size=n_kill, replace=False)
            M = torch.zeros_like(S, dtype=torch.bool)
            for lbl in chosen:
                M |= (S == int(lbl))
            G[M] = 0
            S[M] = 0
            T1[M] = 0
            if T2 is not None:
                T2[M] = 0
            if FLAIR is not None:
                FLAIR[M] = 0
            if CT is not None:
                CT[M] = 0

    # Are we simulating ex vivo (and possibly a bag)
    if mode == 'exvivo':
        M = (D > 0)
        G[M] = 0
        T1[M] = 0
        if T2 is not None:
            T2[M] = 0
        if FLAIR is not None:
            FLAIR[M] = 0
        if CT is not None:
            CT[M] = 0
        if np.random.rand() < bag_prob:  # simulate bag
            bag_scale = bag_scale_min + np.random.rand(1) * (bag_scale_max - bag_scale_min)
            siz_TH_small = np.round(bag_scale * np.array(G.shape)).astype(int).tolist()
            bag_tness = torch.tensor(np.sort(1.0 + 20 * np.random.rand(2)), dtype=dtype, device=device)
            THsmall = bag_tness[0] + (bag_tness[1] - bag_tness[0]) * torch.rand(siz_TH_small, dtype=dtype, device=device)
            TH = myzoom_torch(THsmall, np.array(G.shape) / siz_TH_small, device)
            G[M & (D < TH)] = 99

    if mode == 'exvivo_only_cerebrum':
        M = (ctx.lut_kill_photo[S] == 0)
        G[M] = 0
        T1[M] = 0
        if T2 is not None:
            T2[M] = 0
        if FLAIR is not None:
            FLAIR[M] = 0
        if CT is not None:
            CT[M] = 0
        if np.random.rand() < bag_prob:  # simulate bag
            Dd = torch.minimum(RP, LP)
            Dd[~M] = 0
            bag_scale = bag_scale_min + np.random.rand(1) * (bag_scale_max - bag_scale_min)
            siz_TH_small = np.round(bag_scale * np.array(G.shape)).astype(int).tolist()
            bag_tness = torch.tensor(np.sort(1.0 + (ctx.max_surf_dist_modeled - 1) * np.random.rand(2)), dtype=dtype, device=device)
            THsmall = bag_tness[0] + (bag_tness[1] - bag_tness[0]) * torch.rand(siz_TH_small, dtype=dtype, device=device)
            TH = myzoom_torch(THsmall, np.array(G.shape) / siz_TH_small, device)
            G[M & (Dd < TH)] = 99

    # Photo mode requires killing a bunch of labels
    if mode == 'photo':
        M = ctx.lut_kill_photo[S] == 0
        Mv = ctx.lut_kill_photo_inc_ventricles[S] == 0
        G[Mv] = 0
        T1[M] = 0
        if T2 is not None:
            T2[M] = 0
        if FLAIR is not None:
            FLAIR[M] = 0
        if CT is not None:
            CT[M] = 0

    return {
        'G': G, 'S': S, 'T1': T1, 'T2': T2, 'FLAIR': FLAIR, 'CT': CT,
        'T1dm': T1dm, 'T2dm': T2dm, 'FLAIRdm': FLAIRdm, 'CTdm': CTdm,
        'D': D, 'LP': LP, 'RP': RP, 'LW': LW, 'RW': RW,
        'MNIx': MNIx, 'MNIy': MNIy, 'MNIz': MNIz,
        'xx2': xx2, 'yy2': yy2, 'zz2': zz2,
        'mode': mode, 'resolution': resolution, 'thickness': thickness,
        # kseg (2026-09-30): the crop box and the subject's other contrasts, loaded lazily only when a draw clusters on them
        'bbox': (int(x1), int(x2), int(y1), int(y2), int(z1), int(z2)), 'kseg_src': {'t2': t2, 'flair': flair},
        'stem': ctx.names[d_idx][idx][:-7],
    }


def _intra_grid(ctx, crop_shape, mode, resolution, siz, max_rotation, max_shear, max_scaling,
                nonlin_scale_min, nonlin_scale_max, nonlin_std_max, dtype, device):
    """A fresh affine + nonlinear sampling grid for an ALREADY-CROPPED case: the intra-subject pair of
    paired_pathology_generator (2026-10-27). Same draws as _load_and_prepare_case, centred on the crop (its box was
    cut around the first grid; translation is removed by the training loop's centring anyway), in crop coordinates.
    Corners the first box did not cover read as background through fast_3D_interp_torch's default value."""
    rotations = (2 * max_rotation * np.random.rand(3) - max_rotation) / 180.0 * np.pi
    shears = (2 * max_shear * np.random.rand(3) - max_shear)
    scalings = 1 + (2 * max_scaling * np.random.rand(3) - max_scaling)
    A = torch.tensor(make_affine_matrix(rotations, shears, scalings), dtype=torch.float, device=device)
    c2 = torch.tensor((np.array(crop_shape[0:3]) - 1) / 2, dtype=dtype, device=device)
    nonlin_scale = nonlin_scale_min + np.random.rand(1) * (nonlin_scale_max - nonlin_scale_min)
    siz_F_small = np.round(nonlin_scale * np.array(siz)).astype(int).tolist()
    if mode == 'photo':
        siz_F_small[1] = np.round(siz[1] / resolution[1]).astype(int)
    nonlin_std = nonlin_std_max * np.random.rand()
    Fsmall = nonlin_std * torch.randn([*siz_F_small, 3], dtype=dtype, device=device)
    Field = myzoom_torch(Fsmall, np.array(siz) / siz_F_small, device)
    if mode == 'photo':
        Field[:, :, :, 1] = 0
    xx1 = ctx.xc + Field[:, :, :, 0]
    yy1 = ctx.yc + Field[:, :, :, 1]
    zz1 = ctx.zc + Field[:, :, :, 2]
    xx2 = A[0, 0] * xx1 + A[0, 1] * yy1 + A[0, 2] * zz1 + c2[0]
    yy2 = A[1, 0] * xx1 + A[1, 1] * yy1 + A[1, 2] * zz1 + c2[1]
    zz2 = A[2, 0] * xx1 + A[2, 1] * yy1 + A[2, 2] * zz1 + c2[2]
    return xx2, yy2, zz2


def super_generator(datadir,
              mode_sampler,
              label_list_segmentation,
              n_neutral_labels,
              label_list_normalization,
              flipping=False,
              sides='both',
              labels_to_kill=None,
              list_to_kill_photo=None,
              list_to_kill_photo_inc_ventricles=None,
              random_kill_labels=None,
              random_kill_prob=0.0,
              random_kill_frac=(0.5, 1.0),
              mix_real_images=False,
              minimal_io=True,
              siz=[160, 160, 160],
              max_rotation=15,
              max_shear=0.2,
              max_scaling=0.2,
              nonlin_scale_min=0.03,
              nonlin_scale_max=0.06,
              nonlin_std_max=4,
              bf_scale_min=0.02,
              bf_scale_max=0.04,
              bf_std_min=0.1,
              bf_std_max=0.6,
              bag_scale_min=0.02,
              bag_scale_max=0.08,
              gamma_std=0.1,
              min_noise_std=0.05,
              max_noise_std=0.15,
              bag_prob=0.5,
              deform_one_hots=True,
              bspline_zooming=False,
              dtype=torch.float,
              device='cpu',
              healthy_only=False):

    ctx = _build_generator_context(datadir, label_list_segmentation, n_neutral_labels,
                                    label_list_normalization, list_to_kill_photo,
                                    list_to_kill_photo_inc_ventricles, siz, dtype, device, healthy_only=healthy_only)

    with torch.no_grad():
        while True:
            case = _load_and_prepare_case(ctx, mode_sampler, sides, labels_to_kill,
                                           max_rotation, max_shear, max_scaling,
                                           nonlin_scale_min, nonlin_scale_max, nonlin_std_max,
                                           bag_scale_min, bag_scale_max, bag_prob,
                                           siz, dtype, device,
                                           random_kill_labels=random_kill_labels,
                                           random_kill_prob=random_kill_prob,
                                           random_kill_frac=random_kill_frac,
                                           minimal_io=minimal_io)
            G, S = case['G'], case['S']
            T1, T2, FLAIR, CT = case['T1'], case['T2'], case['FLAIR'], case['CT']
            T1dm, T2dm, FLAIRdm, CTdm = case['T1dm'], case['T2dm'], case['FLAIRdm'], case['CTdm']
            D, LP, RP, LW, RW = case['D'], case['LP'], case['RP'], case['LW'], case['RW']
            MNIx, MNIy, MNIz = case['MNIx'], case['MNIy'], case['MNIz']
            xx2, yy2, zz2 = case['xx2'], case['yy2'], case['zz2']
            mode, resolution, thickness = case['mode'], case['resolution'], case['thickness']

            # Sample Gaussian image
            mus = GMM_MEAN_MIN + (GMM_MEAN_MAX - GMM_MEAN_MIN) * torch.rand(GMM_TABLE_SIZE, dtype=dtype, device=device)
            sigmas = 5 + 20 * torch.rand(256, dtype=dtype, device=device)
            # set the background to zero if photo
            if (mode == 'photo'):
                mus[0] = 0
                sigmas[0] = 0
            elif (np.random.rand() < 0.5):  # actually, let's do this frequently, but not with zero variance!
                mus[0] = 0
            if DARK_CSF_PROB > 0 and np.random.rand() < DARK_CSF_PROB:
                mus = _dark_csf(mus)
            if CONTRAST_T1:
                mus = _contrast_t1(mus)
            if T2_CONTRAST_PROB > 0 and np.random.rand() < T2_CONTRAST_PROB:
                mus = _contrast_t2(mus)
            if CONTRAST_T2:
                mus = _contrast_t2(mus)
            if GMM_SIGMA_REL is not None:
                _lo, _hi = GMM_SIGMA_REL
                sigmas = mus * (_lo + (_hi - _lo) * torch.rand(mus.shape[0], dtype=dtype, device=device))
                if mode == 'photo': sigmas[0] = 0

            #  Crucial bit: partial volume!
            # 1 = lesion, 2 = WM, 3 = GM, 4 = CSF
            v = 0.02 * torch.arange(50).to(device)
            mus[100:150] = mus[1] * (1 - v) + mus[2] * v
            mus[150:200] = mus[2] * (1 - v) + mus[3] * v
            mus[200:250] = mus[3] * (1 - v) + mus[4] * v
            mus[250] = mus[4]
            sigmas[100:150] = torch.sqrt(sigmas[1]**2 * (1 - v) + sigmas[2]**2 * v)
            sigmas[150:200] = torch.sqrt(sigmas[2]**2 * (1 - v) + sigmas[3]**2 * v)
            sigmas[200:250] = torch.sqrt(sigmas[3]**2 * (1 - v) + sigmas[4]**2 * v)
            sigmas[250] = sigmas[4]

            SYN = mus[G] + sigmas[G] * _skew_noise(torch.randn(G.shape, dtype=dtype, device=device), G, mus, GMM_SKEW)
            SYN[SYN < 0] = 0
            SYN /= mus[2]  # normalize by WM
            SYNblur = gaussian_blur_3d(SYN, 0.5 * np.ones(3), device, dtype=dtype)  # cosmetic

            # Make random linear combinations.
            # Anatomy is always a real subject's label map; default False
            # synthesizes contrast via the GMM, so the model never sees real
            # intensities (the contrast-randomization that drives domain
            # generalization). Set True for the original behaviour, where half
            # of all training images blend in the subject's real T1/T2/FLAIR
            # contrast -- trades some contrast randomization for realism.
            # Uses no pathological data either way.
            if mix_real_images and np.random.rand() < 0.5:
                v = torch.rand(4)
                v[2] = 0 if T2 is None else v[2]
                v[3] = 0 if FLAIR is None else v[3]
                v /= torch.sum(v)
                HR = v[0] * SYNblur + v[1] * T1
                if T2 is not None:
                    HR += v[2] * T2
                if FLAIR is not None:
                    HR += v[3] * FLAIR
            else:
                HR = SYNblur

            # deform everything at the same time!
            HRdef, lin_weights = fast_3D_interp_torch(HR, xx2, yy2, zz2, 'linear', device, return_linear_weights=True, dtype=dtype)
            if deform_one_hots:
                Sonehot = ctx.onehotmatrix[ctx.lut[S]]
                SdefOneHot = fast_3D_interp_torch_linear_given_weights(Sonehot, lin_weights, 'linear', device, dtype=dtype)
                # Actually let's make sure background gets extrapolated to p=1 rather than  p=0
                SdefOneHot[..., 0] = fast_3D_interp_torch_linear_given_weights(Sonehot[..., 0], lin_weights, 'linear', device, default_value=1.0, dtype=dtype)
            else:
                Sdef = fast_3D_interp_torch(S, xx2, yy2, zz2, 'nearest', device)
                SdefOneHot = ctx.onehotmatrix[ctx.lut[Sdef.long()]]

            T1def = fast_3D_interp_torch_linear_given_weights(T1, lin_weights, 'linear', device, dtype=dtype)
            T2def = None if T2 is None else fast_3D_interp_torch_linear_given_weights(T2, lin_weights, 'linear', device, dtype=dtype)
            FLAIRdef = None if FLAIR is None else fast_3D_interp_torch_linear_given_weights(FLAIR, lin_weights, 'linear', device, dtype=dtype)
            CTdef = None if CT is None else fast_3D_interp_torch_linear_given_weights(CT, lin_weights, 'linear', device, dtype=dtype)
            MNIxdef = None if MNIx is None else fast_3D_interp_torch_linear_given_weights(MNIx, lin_weights, 'linear', device, dtype=dtype)
            MNIydef = None if MNIy is None else fast_3D_interp_torch_linear_given_weights(MNIy, lin_weights, 'linear', device, dtype=dtype)
            MNIzdef = None if MNIz is None else fast_3D_interp_torch_linear_given_weights(MNIz, lin_weights, 'linear', device, dtype=dtype)
            LPdef = None if LP is None else fast_3D_interp_torch_linear_given_weights(LP, lin_weights, 'linear', device, dtype=dtype, default_value=ctx.max_surf_dist_modeled)
            RPdef = None if RP is None else fast_3D_interp_torch_linear_given_weights(RP, lin_weights, 'linear', device, dtype=dtype, default_value=ctx.max_surf_dist_modeled)
            LWdef = None if LW is None else fast_3D_interp_torch_linear_given_weights(LW, lin_weights, 'linear', device, dtype=dtype, default_value=ctx.max_surf_dist_modeled)
            RWdef = None if RW is None else fast_3D_interp_torch_linear_given_weights(RW, lin_weights, 'linear', device, dtype=dtype, default_value=ctx.max_surf_dist_modeled)
            T1dmdef = None if T1dm is None else fast_3D_interp_torch_linear_given_weights(T1dm, lin_weights, 'linear', device, dtype=dtype)
            T2dmdef = None if T2dm is None else fast_3D_interp_torch_linear_given_weights(T2dm, lin_weights, 'linear', device, dtype=dtype)
            FLAIRdmdef = None if FLAIRdm is None else fast_3D_interp_torch_linear_given_weights(FLAIRdm, lin_weights, 'linear', device, dtype=dtype)
            CTdmdef = None if CTdm is None else fast_3D_interp_torch_linear_given_weights(CTdm, lin_weights, 'linear', device, dtype=dtype)
            Ddef = None if D is None else fast_3D_interp_torch_linear_given_weights(D, lin_weights, 'linear', device, dtype=dtype)

            # Gamma transform
            gamma = torch.tensor(np.exp(gamma_std * np.random.randn(1)[0]), dtype=float, device=device)
            HRgamma = 3.0 * (HRdef / 3.0) ** gamma

            # Bias field
            bf_scale = bf_scale_min + np.random.rand(1) * (bf_scale_max - bf_scale_min)
            siz_BF_small = np.round(bf_scale * np.array(siz)).astype(int).tolist()
            if mode == 'photo':
                siz_BF_small[1] = np.round(siz[1] / resolution[1]).astype(int)
            BFsmall = torch.tensor(bf_std_min + (bf_std_max - bf_std_min) * np.random.rand(1), dtype=dtype, device=device) * torch.randn(siz_BF_small, dtype=dtype, device=device)
            BFlog = myzoom_torch(BFsmall, np.array(siz) / siz_BF_small, device)
            BF = torch.exp(BFlog)
            HRbf = HRgamma * BF

            # Model Resolution
            stds = (0.85 + 0.3 * np.random.rand()) * np.log(5) / np.pi * thickness
            stds[thickness <= 1.0] = 0.0  # no blur if thickness is equal to the resolution of the training data
            HRblur = gaussian_blur_3d(HRbf, stds, device, dtype=dtype)
            newsize = (np.array(siz) / resolution).astype(int)

            factors = np.array(newsize) / np.array(siz)
            delta = (1.0 - factors) / (2.0 * factors)
            vx = np.arange(delta[0], delta[0] + newsize[0] / factors[0], 1 / factors[0])[:newsize[0]]
            vy = np.arange(delta[1], delta[1] + newsize[1] / factors[1], 1 / factors[1])[:newsize[1]]
            vz = np.arange(delta[2], delta[2] + newsize[2] / factors[2], 1 / factors[2])[:newsize[2]]
            II, JJ, KK = np.meshgrid(vx, vy, vz, sparse=False, indexing='ij')
            II = torch.tensor(II, dtype=dtype, device=device)
            JJ = torch.tensor(JJ, dtype=dtype, device=device)
            KK = torch.tensor(KK, dtype=dtype, device=device)

            small = fast_3D_interp_torch(HRblur, II, JJ, KK, 'linear', device, dtype=dtype)
            noise_std = torch.tensor(min_noise_std + (max_noise_std - min_noise_std) * np.random.rand(1), dtype=dtype, device=device)
            smallnoisy = small + noise_std * torch.randn(small.shape, dtype=dtype, device=device)
            smallnoisy[smallnoisy < 0] = 0

            # Back to original resolution
            if bspline_zooming:
                resized = interpol.resize(smallnoisy, shape=siz, anchor='edge', interpolation=3, bound='dct2', prefilter=True)
            else:
                resized = myzoom_torch(smallnoisy, 1 / factors, device)
            maxi = torch.max(resized)
            final = resized / maxi
            SRresidual = (HRbf - resized) / maxi

            # Flip 50% of times
            if (flipping and (np.random.rand() < 0.5)):
                final = torch.flip(final, [0])
                SdefOneHot = torch.flip(SdefOneHot, [0])[:, :, :, ctx.vflip]
                T1def = torch.flip(T1def, [0])
                T2def = None if T2def is None else torch.flip(T2def, [0])
                FLAIRdef = None if FLAIRdef is None else torch.flip(FLAIRdef, [0])
                CTdef = None if CTdef is None else torch.flip(CTdef, [0])
                BFlog = torch.flip(BFlog, [0])
                SRresidual = torch.flip(SRresidual, [0])
                MNIxdef = -torch.flip(MNIxdef, [0])  # careful with switching sign!
                MNIydef = None if MNIydef is None else torch.flip(MNIydef, [0])
                MNIzdef = None if MNIzdef is None else torch.flip(MNIzdef, [0])
                if LPdef is not None or RPdef is not None:
                    aux = None if LPdef is None else torch.flip(LPdef, [0])
                    LPdef = None if RPdef is None else torch.flip(RPdef, [0])
                    RPdef = aux
                if LWdef is not None or RWdef is not None:
                    aux = None if LWdef is None else torch.flip(LWdef, [0])
                    LWdef = None if RWdef is None else torch.flip(RWdef, [0])
                    RWdef = aux
                T1dmdef = None if T1dmdef is None else torch.flip(T1dmdef, [0])
                T2dmdef = None if T2dmdef is None else torch.flip(T2dmdef, [0])
                FLAIRdmdef = None if FLAIRdmdef is None else torch.flip(FLAIRdmdef, [0])
                CTdmdef = None if CTdmdef is None else torch.flip(CTdmdef, [0])
                Ddef = None if Ddef is None else torch.flip(Ddef, [0])

            yield [final, SdefOneHot, T1def, T2def, FLAIRdef, CTdef, BFlog, SRresidual,
                   MNIxdef, MNIydef, MNIzdef,
                   LPdef, RPdef, LWdef, RWdef, T1dmdef, T2dmdef, FLAIRdmdef, CTdmdef, Ddef, mode]


def _mass_effect_jacobian_determinant(grid):
    """Full 3x3 Jacobian determinant of a grid_sample sampling grid, via
    central differences. grid: (1,D,H,W,3), last dim = (x,y,z).

    grid_sample's (x,y,z)<->(D,H,W) axis order is permuted relative to plain
    (i,j,k)->(x,y,z), which flips the sign of the raw determinant for an
    identity grid -- negated here so positive means "no folding" as expected.
    A voxel is locally folded (non-invertible) where this is <= 0.
    """
    gx, gy, gz = grid[0, ..., 0], grid[0, ..., 1], grid[0, ..., 2]
    gx_i, gx_j, gx_k = torch.gradient(gx, dim=(0, 1, 2))
    gy_i, gy_j, gy_k = torch.gradient(gy, dim=(0, 1, 2))
    gz_i, gz_j, gz_k = torch.gradient(gz, dim=(0, 1, 2))
    det = (gx_i * (gy_j * gz_k - gy_k * gz_j)
           - gx_j * (gy_i * gz_k - gy_k * gz_i)
           + gx_k * (gy_i * gz_j - gy_j * gz_i))
    return -det


def _local_thickness_radius(mask, pad=2):
    """Radius of the widest ball that fits anywhere inside `mask`, via a
    Euclidean distance transform -- max distance from an interior voxel to
    the nearest boundary.

    Replaces an equivalent-sphere-from-volume estimate
    ((3*n_vox/(4*pi))**(1/3)), which only sees total voxel count and is
    badly wrong for an elongated shape: a long, thin tract and a compact
    blob of the SAME total volume get the SAME equivalent-sphere radius,
    even though the tract's true cross-sectional width can be several
    times smaller. Verified directly: for a tube of true cross-sectional
    radius 3 and a compact sphere of true radius 9, this function returns
    3.0 and 9.0 exactly (both straight and along a curved, winding path --
    a global covariance/eigenvalue-based estimate was tried first and
    rejected, since winding smears a shape's extent across more axes,
    making a globally-thin tract look artificially closer to isotropic).

    Restricted to a padded bounding box around the mask, not the full
    volume: the distance transform's cost scales with how much space it
    runs over, and everything outside the lesion's own bounding box is
    guaranteed background.
    """
    from scipy.ndimage import distance_transform_edt
    mask_np = mask.detach().cpu().numpy()
    coords = np.argwhere(mask_np)
    if coords.shape[0] < 4:
        return 1.0
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, np.array(mask_np.shape))
    sub = mask_np[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    dist = distance_transform_edt(sub)
    return float(dist.max()) if dist.max() > 1e-6 else 1.0


# Weight on the physically expected polarity direction. Not 1.0: chronic
# blood and calcification are dark on T2, subacute blood and fat are bright on
# T1, so a minority of draws against the expected direction is real coverage
# rather than noise. Same reasoning as the k_rose range.
POLARITY_PRIOR = 0.85

# Lesion INTERIOR texture. Both are DESIGN values (SynthSeg-style: a lesion zone's GMM sigma is capped
# relative to its reference tissue; a multiplicative texture field widens it). A 2026-09-13 comparison
# that called them ~2x too rough against real lesion interiors was RETRACTED (erosion-mask artifact;
# the knob sweep it motivated was null, see memory synthetic-lesion-interiors-too-rough) and no value
# here was ever set from it. Read at call time so pregenerate_validation_set.py can sweep them.
LESION_SIGMA_CAP_MULTIPLE = 2.0   # lesion zone GMM sigma <= this x reference tissue sigma
LESION_TEXTURE_SPREAD_MAX = 1.0   # multiplicative texture field ranges (1-s, 1+s), s ~ U-ish(0, MAX)
# None keeps the cap-only behaviour above. A float SETS the lesion zone's sigma
# to this fraction of the reference tissue's sigma instead of clamping it: a
# cap cannot make interiors smoother than their surround, and measured real
# interiors are (0.62-0.85x the surround's roughness). Halving the cap moved
# synthetic roughness only 1.49 -> 1.32 because the cap rarely binds.
LESION_SIGMA_FRACTION = None
# Structural interior heterogeneity. Both were live defaults, not choices:
# PathologyRandomizer.augment_batch defaults force_rings=True and the call
# below never passed it, so EVERY Local and Tract-Aligned lesion was a
# core/shell ring -- two GMM zones with different means and a sharp internal
# boundary. n_lesion_zones then adds up to MAX_ZONE_LABELS more zones as the
# lesion grows. Real chronic-stroke cavities and whole-tumour masks are far
# more homogeneous inside (measured: smoother than their own surround).
# The clean stream is rendered from the SAME displaced label maps as the
# pathological one (mass effect, atrophy), so the pair differs only in the
# lesion: displacement becomes anatomy, not regression target, and the healthy
# segmentation target is aligned with both images. train.py --displace_clean.
DISPLACE_CLEAN = False

# Choroid plexus as its own TARGET class (2026-09-29, train.py --choroid_target). The generation map keeps the plexus as
# label 9 while the 56-class target merges it into the lateral ventricles; with the flag on, target voxels that are
# generation-label 9 AND inside a left / right ventricle label are relabelled 57 / 58 (appended to the project scheme,
# so every existing id keeps its meaning). Plexus voxels outside a ventricle label (a few percent) are left as they are.
CHOROID_TARGET = False
CHOROID_RAW = (57, 58)          # project ids: left, right
_VENT_L, _VENT_R = (3, 4), (20, 21)


def inject_choroid_target(S, G):
    cp = G == G_CHOROID
    S = torch.where(cp & ((S == _VENT_L[0]) | (S == _VENT_L[1])), torch.full_like(S, CHOROID_RAW[0]), S)
    S = torch.where(cp & ((S == _VENT_R[0]) | (S == _VENT_R[1])), torch.full_like(S, CHOROID_RAW[1]), S)
    return S

# OASIS3 ADRC clinical CDR table (not shipped with the images); see the healthy filter.
OASIS3_CDR_CSV = os.path.join(os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_datasets'), 'OASIS3/OASIS3_UDSb4_cdr.csv')
OASIS3_EXCLUDE_FILE = None   # train.py --oasis_exclude: sessions held out as the healthy evaluation cohort (2026-10-07)
# Elderly healthy cohorts admitted by --healthy; empty = the three young cohorts only.
HEALTHY_EXTRA_DATASETS = ()
# train.py --buckner40. Off by default: Buckner40 has no diagnosis table, so unlike the ADNI3/OASIS3
# opt-ins it cannot be filtered to healthy subjects -- it is admitted whole or not at all.
ALLOW_BUCKNER40 = False
# Datasets dropped from the healthy pool (train.py --exclude_datasets). The base cohorts are assumed
# healthy at dataset level. Buckner40 is NO LONGER in the allowlist at all (2026-10-03): it spans
# young, old and demented adults (Buckner et al. 2004) and so carries diagnosed structural pathology.
EXCLUDE_DATASETS = ()

# Tissue substitution: with probability TISSUE_SUB[mode], the lesion takes
# the GENERATION label of an existing healthy class (CSF 80% / GM 20%) instead
# of a new intensity class, so it renders exactly like that tissue -- healthy
# tissue in the wrong place (a cavity, gliosis, a necrotic core). Vascular:
# the whole lesion; local: an eroded core only. train.py --tissue_sub.
TISSUE_SUB = {}

# Intra-lesion intensity gradient (lever 3). A center->rim ramp on top of the
# existing zero-mean texture field, so a lesion is structurally heterogeneous
# (e.g. a darker necrotic/cavitary core inside a less-affected rim) rather than
# flat +/- the texture noise. Amplitude in (0,1]; 0 = off. Applied to the
# tau-relative target_diff, zero-mean over the mask so it redistributes
# contrast center-vs-rim without changing overall severity. train.py
# --intra_lesion_gradient.
# Per-label GMM means are drawn ~U[GMM_MEAN_MIN, GMM_MEAN_MAX] (SynthSeg's
# 25..225). The floor sets the darkest any tissue can render. Real FLAIR nulls
# CSF to 0.06-0.08 of WM with 1.7-2.4% of brain voxels below 0.1*WM; at a
# floor of 25 the generator's FLAIR-like draws reach 1.2%, and the head reads
# real sulcal CSF as a DARK anomaly -- on Utrecht WMH the dark lobe is 15x the
# bright lobe and takes D_abs from 0.32 to 0.05. A healthy-tissue property:
# applies to every draw, never per pathology mode. train.py --gmm_mean_min.
GMM_MEAN_MIN = 25.0
GMM_MEAN_MAX = 225.0
# Acquisition-grid noise, drawn once per draw as U[NOISE_STD_MIN, NOISE_STD_MAX] and added to the blurred,
# subsampled image BEFORE it is upsampled back. The image is in WM-relative units at that point (SYN /= mus[2]),
# so 0.05-0.15 is 5-15% of the white-matter intensity. SynthSeg's generator carries no additive noise beyond the
# per-class GMM sigma (Billot et al. 2023, brain_generator.py); this band is a PathoPrior addition, and at its
# floor already equals the whole measured within-WM CV of real healthy T1w (Gispert 2004: 5.88% after N3).
# --noise_std MIN MAX overrides (2026-10-05). Upsampling makes the noise spatially correlated at the acquisition
# voxel, which is the source of the mm-scale bright WM blobs the focal measurement found in excess.
NOISE_STD_MIN = 0.05
NOISE_STD_MAX = 0.15
# Per-class GMM sigma as a FRACTION of the class mean (2026-10-05). Default None keeps SynthSeg's absolute
# U[5, 25] on a mean U[25, 225]: at the median WM draw that is 15/125 = 12% per-voxel CV, applied BEFORE the
# resolution blur, so it reaches the network as correlated mm-scale texture -- the measured 11.4% high-passed
# within-WM CV of the renders against 3.9% in the training pool and 5.88% in healthy T1w (Gispert 2004,
# after N3; 4.84% of it noise + partial volume on the phantom). The acquisition-noise band was NOT the source
# (gate 10167841/43: unchanged). With GMM_SIGMA_REL = (lo, hi), sigma[c] = mu[c] * U[lo, hi] per class, drawn
# before the partial-volume ramps so they inherit it. train.py / pregenerate --gmm_sigma_rel.
GMM_SIGMA_REL = None

# Skewed WM intra-class noise (2026-10-08). The pool measurement (wm_focal2, honest cell true|e2|h10|f0.05) says real
# healthy WM is ASYMMETRIC: 5.1 bright focal components per brain (0.25 % of WM, peak z 4.2) against 17.6 dark ones
# (1.27 %, peak z 8.2), while the renders are symmetric (13.0 bright / 7.2 dark, peak z 2-3). Lowering the GMM sigma
# (gate 10168255: 15.3 bright) or the acquisition noise (10167841/43: 16-19) did not move the bright count -- it is
# scale-free under robust z -- so the SHAPE of the draw is the lever, not its amplitude. GMM_SKEW = a reshapes the
# standard-normal per-voxel draw n of the GMM_SKEW_CLASSES voxels (generation label 2 = WM) about the CSF direction
# s = sign(mu_csf - mu_wm), hence contrast-agnostic (dark-heavy on T1-like draws, bright-heavy on T2-like):
#   CSF-ward side (n*s > 0):  n * (1 + a*|n|)    heavy tail (fluid-filled / partial-volume excursions)
#   other side:               n * (1 - a)        light tail
# One transform of the SHARED noise before both renderings, so the deviation target stays exact. Pool gate before any
# fork: wm_focal_gate.sbatch with GATE_XF="--contrast_t1 --gmm_skew a" (a = 0.15, 0.3) against GATE_XF="--contrast_t1".
GMM_SKEW = 0.0
GMM_SKEW_CLASSES = (2,)


def _skew_noise(noise, G, mus, a):
    """Two-piece reshaping of a standard-normal draw on the GMM_SKEW_CLASSES voxels; identity for a <= 0."""
    if a <= 0:
        return noise
    s = float(torch.sign(mus[4] - mus[2]))
    if s == 0:
        return noise
    m = torch.zeros_like(noise, dtype=torch.bool)
    for c in GMM_SKEW_CLASSES:
        m |= (G == c)
    heavy = m & (noise * s > 0)
    light = m & ~heavy
    out = noise.clone()
    out[heavy] = noise[heavy] * (1 + a * noise[heavy].abs())
    out[light] = noise[light] * (1 - a)
    return out

INTRA_LESION_GRADIENT = 0.0

# Dark->fluid intensity (lever 4). Real dark T1 pathology (chronic stroke
# cavity, necrotic core) sits near CSF/fluid intensity, not just tau below its
# surround. When a lesion is drawn DARK, pull its target this fraction of the
# way from the clean intensity toward the local fluid level (estimated from the
# clean image's CSF voxels). 0 = off; 1 = land at fluid. Only affects dark
# draws. train.py --dark_fluid.
DARK_FLUID = 0.0
# Cavity coin (2026-09-29): with probability CAVITY_PROB per dark lesion, the fluid pull is 1.0 (the lesion LANDS at the local fluid
# level) whatever DARK_FLUID says. Diagnostic on the validation set: replacing a synthetic lesion's interior with fluid-level
# intensity halves the head's response inside it (0.53 -> 0.30), while replacing its texture at the same mean changes nothing --
# the head has learned tau-relative offsets, not fluid-filled cavities, which is what a chronic infarct is. train.py --cavity_prob.
CAVITY_PROB = 0.0

# Healthy-tissue heterogeneity (2026-09-21). Real normal-appearing WM/GM is not one
# GMM mean: on FLAIR, mildly hyperintense normal WM ("dirty WM") is what the trained
# head flags as its false positives (94% of the top-1% healthy voxels, 0.45x the lesion
# contrast, 15 mm from any lesion, shared across seeds). With HEALTHY_HETERO = A > 0,
# each healthy class in HEALTHY_HETERO_CLASSES gets its own smooth spectral field
# (zero mean, unit std, the lesion machinery's _universal_spectral_field) scaled by an
# amplitude ~ U(0, A) per image (A = 0.10 recommended: real normal-appearing WM varies 5-10%), multiplied into BOTH the clean and the pathological
# rendering after synthesis, so the regression target (path - clean) is untouched
# and the clean-row loss sees the variation as normal. 0 = off (bit-identical).
HEALTHY_HETERO = 0.0
HEALTHY_HETERO_CLASSES = (2, 3)   # generation labels: WM, GM

# k-means sub-tissue rendering (2026-09-30, train.py --kseg_interp). With probability KSEG_PROB per draw, each healthy
# tissue in KSEG_CLASSES is rendered not as one GMM mean but as k intensity clusters of the subject's OWN T1w (1-D k-means
# on the real intensities inside the label, k ~ U{KSEG_K}). A cluster's synthetic mean is read off the piecewise-linear
# map through the real tissue medians (x) and the drawn GMM means (y): a WM cluster whose real centroid sits 20% of the
# way toward the GM median renders 20% of the way from mus[WM] to mus[GM] under WHATEVER contrast was drawn -- tissue with
# intermediate relaxation lands intermediate under any weighting (dirty WM is darker on T1 and brighter on FLAIR, both
# GM-ward), so the real texture's spatial structure AND ordering survive while no real intensity enters the image. Only a
# label map is derived from the real scan, as the anatomy labels already are (SynthSeg's extracerebral labels are k-means
# clusters of the T1 for the same reason). The per-voxel mean shift (cluster mean - parent mean) is added to BOTH
# renderings before noise/blur, so the regression target (path - clean) is unchanged and the clean-row loss sees the
# variation as normal; no amplitude constant (contrast HEALTHY_HETERO's A), the magnitude follows the real scan. The scan
# clustered is a random convex combination of the subject's T1w / T2w / FLAIR (KSEG_CONTRASTS, those on disk; the same
# weights mix_real_images uses, but here the mixture is only SEGMENTED, never rendered): FLAIR exposes the dirty-WM /
# periventricular / intraventricular texture that T1w barely shows, T2w another view, and a random projection of the three
# is a multi-contrast k-means with the 1-D anchor map kept exact. Mixtures that cancel a tissue gap (T1w and FLAIR order
# GM/WM oppositely) are rejected by a MAD-based separability test and redrawn. T2w/FLAIR crops are read lazily. 0 = off
# (bit-identical, no RNG draw).
KSEG_PROB = 0.0
KSEG_K = (2, 8)                   # k ~ U{KSEG_K[0], ..., KSEG_K[1]} per draw, shared by the tissues
KSEG_CLASSES = (2, 3, 4)          # generation labels: WM, GM, CSF (the anchors of the real -> synthetic map)
KSEG_CONTRASTS = ('t1', 't2', 'flair')   # the real scans that may enter the clustered mixture (those the subject has on disk)
KSEG_MIX_TRIES = 4                # mixture redraws before skipping kseg for the draw (gate below)
# RECOVERABILITY GATE: the original k-means clusters must hold after drawing from the GMMs (contrasts can flip, the labels
# must stay detectable as they were on the clustered image). Per tissue, the rendered image is simulated as
# rendered cluster mean + sigma_c x noise, re-clustered with the same k, and the voxel-wise agreement with the original partition
# (cluster labels matched by rank, a global flip allowed) must be at least what the REAL image gives when perturbed by one unit of
# its own robust noise. Self-calibrating: no observer constant. The test is PER TISSUE: a tissue whose clusters do not survive the
# rendering (GMM sigma swamps the rendered gaps, a folded map, or a partition that was only noise to begin with) is rendered as the
# plain GMM class, the others keep their texture; the mixture is redrawn, then kseg skipped, only when no tissue passes. k is
# adaptive per tissue: the finest clustering (drawn k down to 2) whose labels survive the rendering (metadata k_used).
# No separability gate: T1w, T2w, FLAIR and any mixture are drawn freely (domain randomization).
# Every contrast and every mixture is clustered. Anchors are the tissue medians in whatever order the mixture puts them (CSF between
# WM and GM included); what keeps the real -> synthetic map conditioned is a SLOPE CAP in _kseg_map: a segment shorter than the two
# tissues' pooled robust sigma is widened to that sigma, so one sigma of real texture moves a cluster by at most one tissue contrast
# and a near-isointense pair (WM/GM on FLAIR) cannot amplify noise into the full drawn contrast. The map stays monotone and bounded.
KSEG_ROBUST_SIGMA = 1.4826        # robust sigma = 1.4826 x MAD
KSEG_MIN_VOX = 1000               # a tissue with fewer voxels in the crop is neither clustered nor an anchor
KSEG_T_CLAMP = 0.5                # extrapolation beyond the end anchors, in units of the end segment
KSEG_BINS = 512
KSEG_ITERS = 25

# Data-kind control arms (2026-10-20, train.py --image_source / --anomaly_source): the baselines' KIND of data under our pipeline.
# IMAGE_SOURCE 'real' = HR_clean is the subject's real scan (one contrast per draw, uniform over what it has on disk); ANOMALY_SOURCE
# 'prior' = our lesion painted in through the inverse kseg anchor map (_real_lesion_paint), 'fpi' = DISYRE's 'dag' chain (_fpi_anomaly).
# Rendering-only levers (dark_csf, csf_mix, healthy_hetero) are skipped in real mode. Defaults off: recipe byte-identical.
IMAGE_SOURCE = 'synthetic'
ANOMALY_SOURCE = 'prior'
_REAL_SUFFIX = {'t1': 'T1w.nii', 't2': 'T2w.nii', 'flair': 'FLAIR.nii'}   # pool file per contrast = stem + suffix
# DISYRE 'dag' parameters (get_ag_transforms, preprocessing.py:73-84; D = DISYRE's preprocessing directory)
FPI_PATCH = 64                    # anom_patch_size, DISYRE's 3-D default (fpi.py:21, mask_generation.py:174)
FPI_ALPHA = (0.0, 1.0)            # alpha_range of FPI and BiasCorruption
FPI_CODEBOOK = (0.00565079, 0.38283867, 0.61224216, 0.76920754, 0.9385473)   # preprocessing.py:76, DISYRE's [0,1] frame
FPI_MASK_SCALE = (0.5, 0.75)      # scale_masks: the shape volume is magnified by 1/sc (mask_generation.py:140-143)
FPI_SHAPES_DIR = None             # the 300 binary 128^3 shape volumes (dataset.json); None -> box and sphere only
FPI_BRAIN_Q = 0.877               # DISYRE's unit = the 0.877 quantile of the brain voxels (q98 of a UPD volume with its zeros; data.py:10-12)
FPI_BIAS_PSEUDO = 1e3 * 64        # bias.py:56 pseudo-count 1e3 = 24 % of a 64^2 patch; the same fraction of a 64^3 patch
FPI_CACHE = 64                    # fpi.py:22 size_cache (512 slices); 64^3 float patches, so fewer, per contrast
FPI_VISIBILITY = True             # Rose floor/ceiling on the DISYRE label too (downstream unchanged); False = severities as drawn (--fpi_raw)
REAL_PAINT_T_CLAMP = 2.0          # the paint's extrapolation clamp: KSEG_T_CLAMP 0.5 capped bright T1w lesions at ~WM + 0.5 (WM - GM)
DEG15 = 15 / 360.0 * 2 * math.pi  # DISYRE's rotation range (mask_generation.py:25)


def _contrast_inventory(datasets, names):
    """Per contrast, the pool stems (T1w path minus 'T1w.nii') that have it on disk, printed once as the inventory the launcher
    greps. Expected share = mean over datasets of mean over subjects of 1/n_available: the draw is dataset-uniform, then
    subject-uniform, then contrast-uniform over what the subject has. Stat calls only, no RNG."""
    inv = {c: [] for c in _REAL_SUFFIX}
    share = {c: 0.0 for c in _REAL_SUFFIX}
    parts = []
    for d, files in zip(datasets, names):
        n = {c: 0 for c in _REAL_SUFFIX}
        s = {c: 0.0 for c in _REAL_SUFFIX}
        for t1 in files:
            stem = t1[:-7]
            have = [c for c, suf in _REAL_SUFFIX.items() if c == 't1' or os.path.isfile(stem + suf)]
            for c in have:
                inv[c].append(stem); n[c] += 1; s[c] += 1.0 / len(have)
        parts.append(f"{d} {len(files)} subjects (T1w {n['t1']}, T2w {n['t2']}, FLAIR {n['flair']})")
        for c in share:
            share[c] += s[c] / max(len(files), 1) / max(len(datasets), 1)
    print(f"contrast inventory: {'; '.join(parts)}; total {sum(len(f) for f in names)} (T1w {len(inv['t1'])}, T2w {len(inv['t2'])}, "
          f"FLAIR {len(inv['flair'])}); draw uniform per subject over what exists -> expected shares at dataset-uniform sampling "
          f"T1w {share['t1']:.2f} T2w {share['t2']:.2f} FLAIR {share['flair']:.2f}", flush=True)
    return inv


def _kmeans_1d(x, k, bins=KSEG_BINS, iters=KSEG_ITERS):
    """Lloyd's algorithm on a histogram of the 1-D tensor x: quantile init (deterministic), robust range (0.5-99.5%).
    Returns (sorted centroids (k,), per-element cluster index (x.numel(),) into them)."""
    xs = x[:: max(1, x.numel() // 200000)]
    lo, hi = torch.quantile(xs, torch.tensor([0.005, 0.995], dtype=x.dtype, device=x.device))
    if not bool(hi > lo):
        return x.new_full((1,), float(x.mean())), torch.zeros(x.numel(), dtype=torch.long, device=x.device)
    xc = x.clamp(lo, hi)
    h = torch.histc(xc, bins, float(lo), float(hi))
    centers = lo + (torch.arange(bins, dtype=x.dtype, device=x.device) + 0.5) * (hi - lo) / bins
    k = int(min(k, bins))
    cdf = torch.cumsum(h, 0) / h.sum()
    q = (torch.arange(k, dtype=x.dtype, device=x.device) + 0.5) / k
    c = centers[torch.searchsorted(cdf, q).clamp(0, bins - 1)]
    for _ in range(iters):
        a = torch.argmin((centers[:, None] - c[None, :]).abs(), dim=1)
        w = torch.zeros(k, dtype=x.dtype, device=x.device).index_add_(0, a, h)
        m = torch.zeros(k, dtype=x.dtype, device=x.device).index_add_(0, a, h * centers)
        c = torch.where(w > 0, m / w.clamp_min(1e-12), c)
    c, _ = torch.sort(c)
    keep = torch.ones(k, dtype=torch.bool, device=x.device)
    keep[1:] = c[1:] > c[:-1]      # drop merged (duplicate) centroids
    c = c[keep]
    edges = 0.5 * (c[1:] + c[:-1])
    idx = torch.bucketize(xc, edges)
    return c, idx


def _kseg_map(c, r, s, floor=None, t_clamp=None):
    """Piecewise-linear map real -> synthetic through anchor points (r[i], s[i]), r strictly increasing; linear
    extrapolation with the end segments, clamped to t_clamp (KSEG_T_CLAMP) of a segment beyond each end. floor[i] (the segment's pooled
    robust sigma) caps the slope: a segment narrower than floor[i] is treated as floor[i] wide, so one real sigma moves a
    cluster by at most that segment's synthetic contrast (near-isointense tissue pairs cannot amplify noise)."""
    i = (torch.bucketize(c, r) - 1).clamp(0, r.numel() - 2)
    width = r[i + 1] - r[i]
    if floor is not None:
        width = torch.maximum(width, floor.to(width.dtype).to(width.device)[i])
    t = (c - r[i]) / width
    tc = KSEG_T_CLAMP if t_clamp is None else t_clamp
    t = t.clamp(-tc, 1.0 + tc)
    return s[i] + t * (s[i + 1] - s[i])


def _kseg_volumes(T1, case, device, dtype):
    """The subject's real scans that may be clustered: T1w (in memory) plus the KSEG_CONTRASTS on disk, read as the same crop
    box (minimal_io leaves them unloaded). A crop that does not match the box is dropped."""
    vols = {'t1': T1} if 't1' in KSEG_CONTRASTS else {}
    src = (case or {}).get('kseg_src', {})
    for c in KSEG_CONTRASTS:
        if c == 't1' or not src.get(c):
            continue
        x1, x2, y1, y2, z1, z2 = case['bbox']
        X = torch.squeeze(torch.tensor(np.asarray(nib.load(src[c]).dataobj[x1:x2, y1:y2, z1:z2]), dtype=dtype, device=device))
        if X.shape == T1.shape:
            vols[c] = X
    return vols if vols else {'t1': T1}


def _kseg_anchors(X, masks):
    """Per-tissue median and MAD of X; anchors sorted by median, in whatever order the mixture produces. ok = medians distinct."""
    med = {c: X[m].median() for c, m in masks.items()}
    mad = {c: (X[m] - med[c]).abs().median() for c, m in masks.items()}
    order = sorted(med, key=lambda c: float(med[c]))
    r = torch.stack([med[c] for c in order])
    ok = bool((r[1:] > r[:-1]).all())
    return order, r, ok, [float(mad[c]) for c in order]


def _kseg_segment_floor(mads):
    """Per segment, the pooled robust sigma of its two tissues: the minimum real width a segment is given in _kseg_map."""
    return torch.tensor([KSEG_ROBUST_SIGMA * math.sqrt(0.5 * (mads[i] ** 2 + mads[i + 1] ** 2)) for i in range(len(mads) - 1)])


def _kseg_agreement(idx_a, cent_a, idx_b, cent_b):
    """Voxel-wise agreement of two 1-D clusterings with labels matched by centroid rank; a global flip (contrast inversion) is
    allowed, so the better of the two orientations is returned."""
    ra = torch.argsort(torch.argsort(cent_a))[idx_a]; rb = torch.argsort(torch.argsort(cent_b))[idx_b]
    k = max(int(cent_a.numel()), int(cent_b.numel()))
    same = (ra == rb).float().mean(); flipped = (ra == (k - 1 - rb)).float().mean()
    return float(torch.maximum(same, flipped))


def _kseg_recoverable(X, m, cent, idx, kmu, sigma_c, mad_c, k, dtype):
    """Gate for one tissue: re-cluster the simulated rendering (kmu[idx] + sigma_c * noise) and the real image perturbed by one
    robust-noise unit (X[m] + 1.4826 * MAD * noise) with the same k; pass iff the rendered partition agrees with the original at
    least as well as the perturbed real one does. Returns (ok, agreement_rendered, agreement_reference)."""
    eps = torch.randn(idx.numel(), dtype=dtype, device=idx.device)
    y_r = kmu[idx] + float(sigma_c) * eps
    y_x = X[m].to(dtype) + KSEG_ROBUST_SIGMA * float(mad_c) * eps
    c_r, i_r = _kmeans_1d(y_r, k); c_x, i_x = _kmeans_1d(y_x, k)
    a_r = _kseg_agreement(idx, cent, i_r, c_r); a_x = _kseg_agreement(idx, cent, i_x, c_x)
    return a_r >= a_x, a_r, a_x


def _kseg_shift(G, T1, mus, device, dtype, case=None, sigmas=None):
    """Per-voxel additive mean shift (cluster mean - parent GMM mean) for the tissues in KSEG_CLASSES; zero elsewhere.
    The scan clustered is a random convex combination (mix_real_images' weights) of the subject's real contrasts, each
    standardized to WM median 1; every mixture is accepted (domain randomization), the slope cap in _kseg_map keeps the map
    conditioned; T1w alone only if the mixture's tissue medians coincide. G and the scans must be voxel-aligned (the undisplaced native crop). Returns (shift, metadata) or (None, reason)."""
    k = int(np.random.randint(KSEG_K[0], KSEG_K[1] + 1))
    masks = {}
    for c in KSEG_CLASSES:
        m = G == c
        if int(m.sum()) >= KSEG_MIN_VOX:
            masks[c] = m
    if len(masks) < 2:
        return None, {'kseg': 'skipped: fewer than two anchor tissues in the crop'}
    vols = _kseg_volumes(T1, case, device, dtype)
    wm = masks.get(2, next(iter(masks.values())))
    gm = masks.get(3)
    std, reflected = {}, []
    for c, X in vols.items():
        X = X.to(dtype)
        d = X[wm].median()
        X = X / d if bool(d > 0) else X
        # orient every contrast GM-above-WM (reflect about the WM median, X -> 2 - X) so a mixture cannot cancel the WM-GM
        # axis and GM-ward texture keeps its sign in every contrast (T1w: dirty WM darker; T2w/FLAIR: brighter)
        if gm is not None and bool(X[gm].median() < X[wm].median()):
            X = 2.0 * X[wm].median() - X
            reflected.append(c)
        std[c] = X
    names = sorted(std)
    X = None
    fit = None; last_gate = None
    for _try in range(KSEG_MIX_TRIES):
        # mixture order m ~ U{1..n}: single contrasts, pairs and the full mixture are each drawn exactly, not only strict mixtures
        m_ = int(np.random.randint(1, len(names) + 1)); sel = np.random.choice(len(names), m_, replace=False)
        w = np.zeros(len(names)); w[sel] = np.random.rand(m_) if m_ > 1 else 1.0
        w = w / w.sum()
        X = sum(float(w[i]) * std[n] for i, n in enumerate(names)); weights = {n: round(float(w[i]), 3) for i, n in enumerate(names)}
        order, r, ok, mads = _kseg_anchors(X, masks)
        if not ok:
            continue
        s_ = torch.stack([mus[c] for c in order]).to(dtype)
        floor = _kseg_segment_floor(mads)
        clusters = {c: _kmeans_1d(X[m].to(dtype), k) for c, m in masks.items()}
        kmus = {c: _kseg_map(clusters[c][0], r, s_, floor) for c in masks}
        gate = {}; keep = set(masks); kk = {c: k for c in masks}
        if sigmas is not None:   # recoverability gate, per tissue (see KSEG_MIX_TRIES above); sigmas=None = no gate (diagnostics)
            for c, m in masks.items():
                # adaptive k: the finest clustering, from the drawn k down to 2, whose labels survive the rendering. The number of
                # detectable sub-tissue labels is set by the rendered gap / GMM sigma, so a draw with a large sigma keeps coarser texture
                # rather than none (with a fixed k, WM -- the tissue the lever is for -- survived in ~15% of OASIS3 draws, 2026-10-02).
                okc = False
                for kc in range(k, 1, -1):
                    if kc != kk[c]:
                        clusters[c] = _kmeans_1d(X[m].to(dtype), kc); kmus[c] = _kseg_map(clusters[c][0], r, s_, floor); kk[c] = kc
                    okc, a_r, a_x = _kseg_recoverable(X, m, clusters[c][0], clusters[c][1], kmus[c], sigmas[c], mads[order.index(c)], kc, dtype)
                    if okc:
                        break
                gate[int(c)] = (round(a_r, 3), round(a_x, 3), bool(okc), kk[c])
                if not okc:
                    keep.discard(c)
            last_gate = gate
            if not keep:
                continue
        fit = (X, weights, order, r, mads, floor, clusters, kmus, gate, keep)
        break
    if fit is None:
        return None, {'kseg': 'skipped: no tissue recoverable after rendering', 'contrasts': names, 'last_gate': last_gate}
    X, weights, order, r, mads, floor, clusters, kmus, gate, keep = fit
    capped = [bool(float(r[i + 1] - r[i]) < float(floor[i])) for i in range(len(order) - 1)]
    shift = torch.zeros(G.shape, dtype=dtype, device=device)
    meta = {'k': k, 'k_used': {int(c): int(kk[c]) for c in keep}, 'weights': weights, 'reflected': reflected, 'anchors_real': [float(v) for v in r], 'anchor_mad': mads, 'slope_capped': capped,
            'gate': gate, 'textured': sorted(int(c) for c in keep), 'anchor_order': order, 'tissues': {}}
    for c, m in masks.items():
        if c not in keep:
            continue          # clusters not recoverable after rendering: this tissue stays the plain GMM class
        cent, idx = clusters[c]
        kmu = kmus[c]
        shift[m] = kmu[idx] - mus[c]
        frac = torch.bincount(idx, minlength=cent.numel()).to(dtype) / idx.numel()
        meta['tissues'][int(c)] = {'n_clusters': int(cent.numel()),
                                   'shift_range_wm': (float((kmu.min() - mus[c]) / mus[2]), float((kmu.max() - mus[c]) / mus[2])),
                                   'centroids': [round(float(v), 3) for v in cent], 'frac': [round(float(v), 3) for v in frac]}
    return shift, meta


def _real_lesion_paint(X_w, SYN_path, SYN_clean, X, G_native, L, mus, sigmas, device, dtype):
    """FAE-kind pathological image (2026-10-20): the GMM-rendered lesion (SYN_path, WM-relative) mapped into the real frame by the
    INVERSE kseg anchor map (drawn WM/GM/CSF means <-> real tissue medians of the undisplaced pair X, G_native; slope floor = pooled
    GMM sigma, mirroring _kseg_segment_floor; extrapolation clamped at REAL_PAINT_T_CLAMP segments, hit rate in meta), composited
    into the warped scan X_w under L with _synth's PV blur. Fallback below two anchor tissues: the additive WM-relative deviation."""
    meta = {'n_vox': int(L.sum())}
    if not bool(L.any()):
        return X_w, meta
    masks = {c: G_native == c for c in KSEG_CLASSES}
    masks = {c: m for c, m in masks.items() if int(m.sum()) >= KSEG_MIN_VOX}
    paint = None
    if len(masks) >= 2:
        order, r, _ok, _ = _kseg_anchors(X, masks)
        s = torch.stack([mus[c] for c in order]).to(dtype) / mus[2]
        sig = torch.stack([sigmas[c] for c in order]).to(dtype) / mus[2]
        idx = torch.argsort(s)
        s, r, sig = s[idx], r[idx].to(dtype), sig[idx]
        keep = torch.ones(s.numel(), dtype=torch.bool, device=s.device)
        keep[1:] = s[1:] > s[:-1]          # merged drawn means: one anchor
        s, r, sig = s[keep], r[keep], sig[keep]
        if s.numel() >= 2:
            floor = torch.sqrt(0.5 * (sig[1:] ** 2 + sig[:-1] ** 2))
            v = SYN_path[L].to(dtype)
            paint = _kseg_map(v, s, r, floor, REAL_PAINT_T_CLAMP).clamp_min(0)
            w = torch.maximum(s[1:] - s[:-1], floor)   # the end widths _kseg_map extrapolates with
            lo, hi = s[0] - REAL_PAINT_T_CLAMP * w[0], s[-2] + (1.0 + REAL_PAINT_T_CLAMP) * w[-1]
            meta.update({'anchor_order': [int(order[i]) for i, k in zip(idx.tolist(), keep.tolist()) if k],
                         'anchors_syn': [round(float(v), 3) for v in s], 'anchors_real': [round(float(v), 3) for v in r],
                         'slope_floor': [round(float(v), 3) for v in floor],
                         'clamp_hit': round(float(((v < lo) | (v > hi)).float().mean()), 4)})   # pool-level share for the note
    if paint is None:
        paint = (X_w[L] + SYN_path[L] - SYN_clean[L]).clamp_min(0)
        meta['fallback'] = 'additive'
    D = torch.zeros_like(X_w)
    D[L] = paint - X_w[L]
    return (X_w + gaussian_blur_3d(D, 0.5 * np.ones(3), device, dtype=dtype)).clamp_min(0), meta


# DISYRE-kind anomaly (2026-10-20): the 'dag' chain of D/{mask_generation,fpi,bias}.py (D = DISYRE's preprocessing directory;
# 2-D port anomalies.py) in torch, 3-D as DISYRE's defaults; each formula names its source line. Deviations
# (3-D or our WM-median frame): trilinear mask resampling, pseudo-count scaled to a 64^3 patch, [0,1] frame = x / q_0.877(brain)
# clipped at the brain max, brain mask = S > 0 minus extracerebral where the reference tests image > 0.
_FPI_SHAPES = {}


def _fpi_shape_list():
    """The shape volumes of FPI_SHAPES_DIR (dataset.json 'training'), each loaded once on first use and kept bit-packed
    (RandomMask, mask_generation.py:159-163; DataLoader3D infinite = uniform with replacement)."""
    if 'list' not in _FPI_SHAPES:
        import json
        ds = json.load(open(os.path.join(FPI_SHAPES_DIR, 'dataset.json')))
        _FPI_SHAPES['list'] = [os.path.join(FPI_SHAPES_DIR, os.path.basename(it['label'])) for it in ds['training']]
        shp = nib.load(_FPI_SHAPES['list'][0]).shape
        print(f"fpi shapes: {len(_FPI_SHAPES['list'])} volumes {list(shp)}, "
              f"{len(_FPI_SHAPES['list']) * float(np.prod(shp)) / 8 / 2 ** 20:.0f} MB per process once all are drawn (bit-packed)", flush=True)
    return _FPI_SHAPES['list']


def _fpi_shape_volume(i):
    if i not in _FPI_SHAPES:
        v = np.asarray(nib.load(_fpi_shape_list()[i]).dataobj) > 0
        _FPI_SHAPES[i] = (np.packbits(v.ravel()), v.shape)   # unpacked, 300 x 128^3 bools would hold 630 MB per process
    bits, shape = _FPI_SHAPES[i]
    return np.unpackbits(bits, count=int(np.prod(shape))).reshape(shape).astype(bool)


def _fpi_blur(x, sigma, device, dtype):
    """scipy.ndimage.gaussian_filter as batchgenerators' augment_gaussian_blur calls it: truncate 4 sigma, 'reflect' border
    (half-sample symmetric, d c b a | a b c d; torch's 'reflect' is scipy's 'mirror', so the pad is built by hand)."""
    lw = int(4.0 * sigma + 0.5)
    t = torch.arange(-lw, lw + 1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (t / sigma) ** 2); k = k / k.sum()
    y = x
    for d in range(3):
        y = torch.cat([y.narrow(d, 0, lw).flip(d), y, y.narrow(d, y.shape[d] - lw, lw).flip(d)], d)
    y = y[None, None]
    y = conv3d(y, k[None, None, :, None, None]); y = conv3d(y, k[None, None, None, :, None]); y = conv3d(y, k[None, None, None, None, :])
    return y[0, 0]


def _fpi_resample(vol, P, angles, sc, device, dtype):
    """augment_spatial_2 on a mask (batchgenerators_custom.py:104-146): the zero-centered P^3 grid rotated by three angles,
    scaled by sc (the sampled extent, so the content is magnified by 1/sc), re-centered on the volume, 'nearest' border."""
    ax = torch.arange(P, dtype=dtype, device=device) - (P - 1) / 2.0
    g = torch.stack(torch.meshgrid(ax, ax, ax, indexing='ij'), -1)              # (P,P,P,3) voxel offsets
    R = torch.eye(3, dtype=dtype, device=device)
    for a, (i, j) in zip(angles, ((1, 2), (0, 2), (0, 1))):                      # rotate_coords_3d: x, then y, then z
        c, s = math.cos(a), math.sin(a)
        M = torch.eye(3, dtype=dtype, device=device); M[i, i] = c; M[i, j] = -s; M[j, i] = s; M[j, j] = c
        R = M @ R
    g = (g @ R.T) * sc + torch.tensor([(n - 1) / 2.0 for n in vol.shape], dtype=dtype, device=device)
    g = g / torch.tensor([n - 1 for n in vol.shape], dtype=dtype, device=device) * 2 - 1   # align_corners=True convention
    out = F.grid_sample(vol[None, None].to(dtype), g[None][..., [2, 1, 0]], mode='bilinear', padding_mode='border', align_corners=True)
    return out[0, 0]


def _fpi_shape_patch(P, device, dtype):
    """CreateRandomShape (mask_generation.py:213-289) with DISYRE's arguments: kind ~ 1:1:1 box / sphere / shape volume (volume
    absent without FPI_SHAPES_DIR), spatial transform, then random_smooth; returns (P^3 soft mask in [0,1], metadata)."""
    r = np.random.rand
    kinds = ['box', 'sphere'] + (['volume'] if FPI_SHAPES_DIR else [])
    kind = kinds[np.random.randint(len(kinds))]
    meta = {'fpi_kind': kind}
    if kind == 'box':                                                            # RandomSquare :64-105 (3-D branch :96-99)
        size = [int(np.random.randint(10, P - 20 + 1)) for _ in range(3)]       # random.randint is inclusive
        ctr = [int(np.random.randint(a // 2, P - a // 2 + 1)) for a in size]
        m = torch.zeros((P, P, P), dtype=dtype, device=device)
        m[ctr[0] - size[0] // 2:ctr[0] + size[0] // 2, ctr[1] - size[1] // 2:ctr[1] + size[1] // 2, ctr[2] - size[2] // 2:ctr[2] + size[2] // 2] = 1
        meta['size'] = size
        if r() < 0.5:                                                            # spatial_prob .5 (:227), scale (1,1) (:18)
            angles = [float(np.random.uniform(-DEG15, DEG15)) for _ in range(3)]
            m = (_fpi_resample(m, P, angles, 1.0, device, dtype) > 0.5).to(dtype)   # :43 re-threshold
            meta['angles_deg'] = [round(a * 180 / math.pi, 1) for a in angles]
    elif kind == 'sphere':                                                       # RandomSphere :107-132, spatial_prob 0
        size = int(np.random.randint(10, P - 20 + 1))
        ctr = torch.tensor([int(np.random.randint(size // 2 + 10, P - size // 2 - 10 + 1)) for _ in range(3)], dtype=dtype, device=device)
        ax = torch.linspace(0, P, P, dtype=dtype, device=device)                 # :118 linspace(0, P, P): spacing P/(P-1), kept
        mesh = torch.stack(torch.meshgrid(ax, ax, ax, indexing='ij'), 0)
        m = (((mesh - ctr[:, None, None, None]) ** 2).sum(0) <= (size // 2) ** 2).to(dtype)
        meta['size'] = size
    else:                                                                        # RandomMask :135-168 + spatial_prob 1, scale_masks
        i = int(np.random.randint(len(_fpi_shape_list())))
        vol = torch.as_tensor(_fpi_shape_volume(i), device=device)
        lo, hi = FPI_MASK_SCALE                                                  # batchgenerators_custom.py:127-132 (scale[0] < 1)
        sc = float(np.random.uniform(lo, 1.0)) if r() < 0.5 else float(np.random.uniform(max(lo, 1.0), hi))
        angles = [float(np.random.uniform(-DEG15, DEG15)) for _ in range(3)]
        m = (_fpi_resample(vol, P, angles, sc, device, dtype) > 0.5).to(dtype)
        meta.update({'shape_index': i, 'scale': round(sc, 3), 'angles_deg': [round(a * 180 / math.pi, 1) for a in angles]})
    sigma = 0
    if r() <= 0.7:                                                               # random_smooth :46-51: p .7, int sigma of U(0,5)
        sigma = int(np.random.uniform(0, 5))                                     # get_range_val casts to the range's int type
        if sigma > 0:
            m = _fpi_blur(m, float(sigma), device, dtype)
    meta['smooth_sigma'] = sigma
    return m, meta


def _fpi_crop(X, brain, P, nz):
    """CropForeground + RandomCropTransform(P) (fpi.py:57-59): the brain bbox (nz = the brain voxel indices) of the brain-masked
    scan, then a random P^3 crop of it, zero-padded where the bbox is smaller (batchgenerators crop semantics, anomalies.py:75-93)."""
    if nz.numel() == 0:
        return torch.zeros((P, P, P), dtype=X.dtype, device=X.device)
    lo, hi = nz.min(0).values.tolist(), (nz.max(0).values + 1).tolist()
    Xb = (X * brain)[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    out = torch.zeros((P, P, P), dtype=X.dtype, device=X.device)
    src, dst = [], []
    for d, n in enumerate(Xb.shape):
        lb = int(np.random.randint(0, n - P)) if n - P > 0 else (n - P) // 2
        src.append(slice(max(0, lb), min(lb + P, n))); dst.append(slice(max(0, -lb), max(0, -lb) + min(lb + P, n) - max(0, lb)))
    out[tuple(dst)] = Xb[tuple(src)]
    return out


def _fpi_anomaly(X, brain, cache, contrast, stem, device, dtype):
    """One DISYRE 'dag' draw on the real scan X (WM-median frame; brain = foreground): location -> shape -> FPI texture from a
    same-contrast patch out of DISYRE's FIFO of recent crops (fpi.py:68-99; self-patch only during warm-up) -> v2 bias under the
    same shape (preprocessing.py:80-101). Returns (X_path, soft shape zero outside the brain, metadata)."""
    P = int(FPI_PATCH)
    size = list(X.shape)
    if any(n < P for n in size):
        raise ValueError(f"FPI_PATCH {P} exceeds the native crop {size}; lower --fpi_patch or raise siz")
    # GetRandomLocation (mask_generation.py:187-199): a brain voxel >= P//2 from every border, else uniform in the margin
    nz = brain.nonzero()
    lo = torch.tensor([P // 2] * 3, device=nz.device); hi = torch.tensor([n - P // 2 for n in size], device=nz.device)
    ok = ((nz >= lo) & (nz < hi)).all(1)
    if bool(ok.any()):
        c = nz[ok][int(np.random.randint(int(ok.sum())))].tolist()
    else:
        c = [int(np.random.randint(P // 2, n - P // 2 + 1)) for n in size]
    win = tuple(slice(ci - P // 2, ci + P // 2) for ci in c)
    patch, meta = _fpi_shape_patch(P, device, dtype)
    shape = torch.zeros_like(X)
    shape[win] = patch                                                           # CreateRandomShape :274-277
    shape[~brain] = 0                                                            # no_mask_in_background :283-284
    # FPI (fpi.py:81-158)
    q = cache.setdefault(contrast, [])
    pushed = 0
    while len(q) < FPI_CACHE and pushed < 2:                                     # :93-99 warm-up, two patches per call
        q.append((_fpi_crop(X, brain, P, nz), stem)); pushed += 1
    fp, donor = q.pop(int(np.random.randint(len(q))))                            # :81
    alpha_t = float(np.random.uniform(*FPI_ALPHA))                               # :82
    fp_big = torch.zeros_like(X)
    fp_big[win] = fp                                                             # :116-119
    m = shape > 0
    if bool(m.any()):                                                            # normalize_fp 'minmax' :143-148, then clip :150
        mi, mf = X[m], fp_big[m]
        fp_big = (fp_big - mf.min()) / (mf.max() - mf.min() + 1e-3)
        fp_big = (fp_big * (mi.max() - mi.min()) + mi.min()).clamp_min(0)      # the upper clip is DISYRE's [0,1] frame: none here
    out = X * (1 - shape * alpha_t) + fp_big * shape * alpha_t                   # :154
    if torch.equal(out, X):
        alpha_t = 0.0                                                            # :157-158
    # BiasCorruption (bias.py:36-93) in DISYRE units d = x / q_0.877(brain) (data.py:10-12)
    v = X[brain].float() if bool(brain.any()) else X.flatten().float()
    if v.numel() > 4_000_000:   # torch.quantile input cap; a random subsample keeps the quantile unbiased
        v = v[torch.randint(v.numel(), (4_000_000,), device=v.device)]
    qb = torch.quantile(v, FPI_BRAIN_Q)
    qb = qb.to(dtype).clamp_min(1e-6)
    d = out / qb
    cb = torch.tensor(FPI_CODEBOOK, dtype=dtype, device=device)
    idx = torch.bucketize(d.contiguous(), 0.5 * (cb[1:] + cb[:-1]))             # nearest codebook entry (scipy vq, :6-9, :50)
    n_k = torch.bincount(idx[m], minlength=cb.numel()).to(dtype)                 # :53-55
    sel = torch.rand(cb.numel(), device=device) < n_k / (n_k.sum() + FPI_BIAS_PSEUDO)   # :56
    if not bool(sel.any()):
        sel[:] = True                                                            # :58
    ind = _fpi_blur(sel[idx].to(dtype), 1.0, device, dtype)                      # :59-63 indicator over the whole image, sigma 1
    B = shape * ind                                                              # :81
    alpha_b = float(np.random.uniform(*FPI_ALPHA)); sign = 1.0 if np.random.rand() < 0.5 else -1.0   # :46-47
    dmax = d[brain].max() if bool(brain.any()) else d.max()
    d_c = torch.where(B > 0, (d * (1 - B) + (d + alpha_b * sign) * B).clamp(0, dmax), d)   # :82-83, clip top = brain max
    X_path = (out + (d_c - d) * qb).clamp_min(0)                                # :83 clip at 0 (rounding can dip below)
    meta.update({'alpha_texture': round(alpha_t, 4), 'alpha_bias': round(float((d_c - d).abs().max()), 4),   # :90
                 'bias_sign': sign, 'bias_classes': [int(i) for i in sel.nonzero().flatten().tolist()],
                 'donor': os.path.basename(donor), 'self_patch': donor == stem, 'center': c,
                 'n_label': int(((shape >= 0.5) & brain).sum())})
    return X_path, shape, meta


def _healthy_hetero_field(G, device, dtype):
    """1 + a*z per healthy class, z a smooth zero-mean unit-std spectral field; clamped so
    a tissue never darkens below half its mean."""
    from pathology_primitives import PathologyRandomizer
    F = torch.ones(G.shape, dtype=dtype, device=device)
    aug = PathologyRandomizer((1, 1, *G.shape), device=device)
    for c in HEALTHY_HETERO_CLASSES:
        m = G == c
        if not bool(m.any()):
            continue
        a = HEALTHY_HETERO * float(np.random.rand())
        # max_freq 0.06 cycles/voxel: structure no finer than ~17 voxels, coarser than any lesion;
        # z clamped to +-2 sigma so the modulation never exceeds 2A (A = 0.10 -> at most +-20%).
        z = torch.clamp(aug._universal_spectral_field(1, max_freq=0.06)[0, 0].to(dtype), -2.0, 2.0)
        F[m] = 1.0 + a * z[m]
    return F


# Dark-CSF contrast prior. In generation_labels.nii CSF is ONE class (G = 4,
# ventricles and sulci alike; WM = 2, GM = 3; verified on HCP.sub-001,
# 2026-09-17), so its GMM mean is a single draw from U[GMM_MEAN_MIN, GMM_MEAN_MAX]
# independent of WM: only ~8.6% of draws render CSF darker than WM - 0.3 in the
# model frame, and ~1.7% of lesion components sit in the real FLAIR/T1
# configuration (bright/dark lesion AND very dark CSF). Real T1 and FLAIR both
# have CSF/WM ~ 0.1-0.4: derivable from relaxometry (3 T T1: WM ~0.83 s, GM ~1.33 s, CSF ~4 s; Wansapura
# et al., JMRI 1999;9:531; Stanisz et al., MRM 2005;54:507) through the MPRAGE signal equation (Mugler &
# Brookeman, MRM 1990;15:152; Deichmann et al., NeuroImage 2000;12:112), and FLAIR nulls CSF by design
# (Hajnal et al., JCAT 1992;16:841). A physics range, not a measurement on any test set (2026-10-05).
# With prob DARK_CSF_PROB per image the CSF mean is
# re-drawn as mus[2] * U[DARK_CSF_RANGE] before the PV ramps, so the GM->CSF
# ramp (200-250) inherits it. Sigmas untouched. train.py --dark_csf.
DARK_CSF_PROB = 0.0
DARK_CSF_RANGE = (0.10, 0.40)
# CSF intra-class heterogeneity (2026-10-07). Generation label 4 is ONE CSF class (ventricles and sulci) rendered as a single
# Gaussian, so the healthy prior's CSF is more uniform than real CSF, and the detector's false positives on ATLAS T1w and BraTS
# T2w are the CSF intensity extremes (ventricle interior, sulcal CSF; leg_failure_signature 2026-10-07). Real CSF is not one
# Gaussian -- literature, never a test-set measurement:
#   * SPM's unified segmentation models CSF with TWO Gaussians by default (Ashburner & Friston 2005, NeuroImage 26:839; SPM12 defaults).
#   * FLAIR: intraventricular CSF pulsation artefact (hyperintense CSF) in 72 of 100 consecutive NORMAL studies at 1.5 T -- fourth
#     ventricle 58 %, third 47 %, lateral 13 % (Bakshi et al. 2000, AJNR 21:503); at 3 T in 86 % of 200 unremarkable studies (fourth
#     73 %, third 42 %, lateral 39 %; Lavdas et al. 2014, Clin Imaging 38:85).
#   * T2w: the CSF flow-void sign (hypointense pulsatile CSF) in the aqueduct, fourth and third ventricles and foramina of Monro
#     (100 / 96 / 70 / 33 % with ventriculomegaly; Sherman et al. 1987, AJNR 8:1025); "striking heterogeneity in the appearance of CSF".
#   * Choroid plexus occupies ~3.2 +/- 1.4 mL of the lateral ventricles in 98 adults (Eisma et al. 2024, Fluids Barriers CNS 21:21),
#     i.e. of the order of a tenth of the cavity, at an intensity between CSF and GM (see G_CHOROID above).
#   * Relaxometry: in-vivo T1 varies 6.6 % within CSF against 1.8 % within WM (Jordanova et al. 2025, JACMP 26:e70289).
# Model: with probability CSF_MIX_PROB per draw, a smooth spectral field selects a fraction U[CSF_MIX_FRAC] of the CSF voxels and
# renders them as a SECOND component whose mean lies U[CSF_MIX_PULL] of the way from the CSF mean toward the GM mean (toward tissue,
# whatever the contrast: FLAIR pulsation shares GM/WM intensity, T2 flow voids lose signal toward tissue, T1 plexus sits near GM).
# The same field multiplies the clean and the pathological rendering, so the deviation target is untouched and the network learns
# that CSF heterogeneity is normal. Sigmas untouched. train.py --csf_mix.
CSF_MIX_PROB = 0.0
CSF_MIX_FRAC = (0.05, 0.30)
CSF_MIX_PULL = (0.25, 1.00)
CSF_MIX_MAX_FREQ = 0.10   # cycles/voxel: blobs no finer than ~10 voxels, the scale of flow artefacts and plexus, coarser than most lesions
G_WM, G_GM, G_CSF = 2, 3, 4
# Choroid plexus (2026-09-29): generation label 9 is a ~1 mL structure almost entirely inside the lateral ventricles
# (HCP.sub-001: 957 of 1025 voxels within brainseg 3/20), i.e. the choroid plexus kept as its own Gaussian while the
# 56-class target merges it into the ventricle. The agnostic draw leaves its mean free (U[GMM_MEAN_MIN, GMM_MEAN_MAX]);
# the contrast-realistic modes below pin its ORDERING relative to GM and CSF, from the literature, never from test data:
#   T1w:   between CSF and grey matter, nearer GM -- "hyperintense compared to CSF" (Tireli et al. 2025, Fluids Barriers CNS 22:92),
#          "normo-to-mildly hypointense choroid plexus signal" on MPRAGE (Eisma et al. 2024, Fluids Barriers CNS 21), "medium- to
#          low-intensity signal compared with white matter" (Hinshaw et al. 1988, AJNR 9:483); quantitatively T1(CP) 2.33 s vs
#          T1(GM) 1.85 s at 3 T (Zhao et al. 2020, Fluids Barriers CNS 17:58) and 2504 vs 1726 ms at 7 T (Senthil et al. 2026,
#          J Neuroimaging 36:e70132), i.e. ~20 % of the way from GM toward CSF.
#   FLAIR: hyperintense against the suppressed CSF, sharing intensities with GM/WM -- "CPs appear hyperintense on FLAIR images and
#          the signal of the CSF is suppressed" (Storelli et al. 2023, JMRI; Preziosa et al. 2024/2025).
#   T2w:   "isointense to surrounding brain tissues" (Tireli et al. 2025), "relatively intermediate signal intensity ... highly
#          variable", bright glomera in 39.5 % of 167 patients (Hinshaw et al. 1988): iso to GM with a bright tail toward CSF.
G_CHOROID = 9
CHOROID_T1_RANGE = (0.75, 1.00)    # _contrast_t1: choroid = GM x U[range]  (mildly hypo- to iso-intense to GM, above CSF)
CHOROID_DARKCSF_RANGE = (0.75, 1.20)   # _dark_csf covers T1-like AND FLAIR-like draws: iso to GM with both tails
CHOROID_T2_RANGE = (0.00, 0.50)    # _contrast_t2: choroid = GM + U[range] x (CSF - GM)  (iso to GM, bright tail toward CSF)


def _pin_choroid(mus, mode):
    r = lambda a, b: a + (b - a) * float(np.random.rand())
    if mode == "t2":
        mus[G_CHOROID] = mus[G_GM] + r(*CHOROID_T2_RANGE) * (mus[G_CSF] - mus[G_GM])
    else:
        mus[G_CHOROID] = mus[G_GM] * r(*(CHOROID_T1_RANGE if mode == "t1" else CHOROID_DARKCSF_RANGE))
    return mus


def _csf_mix_field(G, mus, device, dtype):
    """Multiplicative field for the second CSF component (see CSF_MIX_* above): 1 everywhere except on a fraction of the CSF
    voxels, selected by a smooth spectral field, where it equals (second-component mean) / (CSF mean)."""
    from pathology_primitives import PathologyRandomizer
    F = torch.ones(G.shape, dtype=dtype, device=device)
    m = G == G_CSF
    n = int(m.sum())
    if n < 100:
        return F
    r = lambda a, b: a + (b - a) * float(np.random.rand())
    frac, pull = r(*CSF_MIX_FRAC), r(*CSF_MIX_PULL)
    aug = PathologyRandomizer((1, 1, *G.shape), device=device)
    z = aug._universal_spectral_field(1, max_freq=CSF_MIX_MAX_FREQ)[0, 0].to(dtype)
    v = z[m].float()
    if v.numel() > 100_000:   # torch.quantile input cap; a random subsample keeps the threshold unbiased
        v = v[torch.randint(v.numel(), (100_000,), device=v.device)]
    thr = torch.quantile(v, 1.0 - frac)
    sel = m & (z > thr)
    mu_csf, mu_gm = float(mus[G_CSF]), float(mus[G_GM])
    F[sel] = (mu_csf + pull * (mu_gm - mu_csf)) / max(mu_csf, 1e-3)
    return F


def _dark_csf(mus):
    lo, hi = DARK_CSF_RANGE
    mus[G_CSF] = mus[G_WM] * (lo + (hi - lo) * float(np.random.rand()))
    return _pin_choroid(mus, "darkcsf")


# T1-contrast specialization (ablation, 2026-09-25): instead of randomizing every
# tissue mean (contrast-agnostic default), render a T1-realistic ordering
# WM > GM > CSF with clinical ratios (GM/WM ~ 0.55-0.80, CSF/WM ~ 0.10-0.40; same relaxometry /
# MPRAGE-equation derivation as DARK_CSF_RANGE above, no test-set measurement),
# so the model specializes to T1 the way DISYRE/dmprior train per modality.
# Bias field, noise, gamma and resolution randomization are untouched. Lesion
# mean (G=1) and other structure labels keep their draws. train.py --contrast_t1.
CONTRAST_T1 = False


def _contrast_t1(mus):
    r = lambda a, b: a + (b - a) * float(np.random.rand())
    wm = GMM_MEAN_MAX * r(0.65, 1.0)
    mus[G_WM] = wm
    mus[G_GM] = wm * r(0.55, 0.80)
    mus[G_CSF] = wm * r(0.10, 0.40)
    return _pin_choroid(mus, "t1")


# T2-realistic contrast prior. The base GMM draws each tissue mean independently,
# so the fully T2-inverted ordering (CSF > GM > WM) with realistic ratios and a
# separable bright lesion occurs in only ~3-8% of images (measured by sampling
# the draw, not on any test set) -- the model is under-exposed to the regime where
# a normal structure (unattenuated bright CSF) shares the pathology's appearance.
# On T2w CSF is NOT fluid-attenuated, so it is the brightest tissue: the T2w tumor
# failure (AUC ~0.52) is that bright CSF out-responds bright edema. Ratios from MR
# physics / literature (T2w CSF/WM ~2-4, GM/WM ~1.1-1.4, WM darkest), never from
# BraTS. Lesion mean (G=1) keeps its own draw so lesion polarity is still handled
# by the target_diff machinery. Two entry points, mirroring dark_csf + contrast_t1:
#   T2_CONTRAST_PROB (per-image enrichment during training; train.py --t2_prob)
#   CONTRAST_T2      (forced, for the in-domain gate / a T2 specialization arm;
#                     train.py / pregenerate_validation_set.py --contrast_t2)
CONTRAST_T2 = False
T2_CONTRAST_PROB = 0.0


def _contrast_t2(mus):
    r = lambda a, b: a + (b - a) * float(np.random.rand())
    wm = GMM_MEAN_MAX * r(0.20, 0.45)   # WM darkest; leaves headroom for CSF up to 4x WM
    mus[G_WM] = wm
    mus[G_GM] = wm * r(1.10, 1.40)
    mus[G_CSF] = wm * r(2.00, 4.00)
    return _pin_choroid(mus, "t2")

# Mode-aware lesion polarity: P(bright) per primary mode and rendered contrast
# ('t1' = WM brighter than GM, 't2' = the reverse). Empty = current behaviour
# (floor branch: contrast prior; GMM branch: coin flip). train.py --mode_polarity.
# Chronic infarcts and non-enhancing tumours are dark on T1; WMH/infarcts are
# bright on FLAIR/T2.
POLARITY_BY_MODE = {}

# Concentric intensity zones: PathologyRandomizer already builds a core and a
# shell for ring lesions (force_rings), but the GMM zones were cut from a
# spectral field and the ring was discarded. With this on, a ring lesion gets
# zone 1 = core, zone 2 = shell, and the two zones draw their signs
# independently (dark necrotic core inside a bright rim, or the reverse)
# instead of the one-sign-per-lesion rule. train.py --concentric_zones.
CONCENTRIC_ZONES = False
# Nested compartments for LOCAL lesions (2026-10-01, train.py --local_compartments p): with probability p a Local-Pathology
# lesion is rendered as core / rim / edema -- the ring core (or an eroded core) and the rim draw their signs independently as
# concentric zones do, and an EDEMA margin is grown around the lesion through healthy WM/GM (generation labels 2/3), added to
# the lesion label (whole-tumour convention) and rendered with the rim's sign at a fraction of its deviation. Motivation: the
# residual is lower INSIDE real lesions than in the ring outside them on every leg; a large synthetic lesion is one GMM texture,
# so the network has never had a heterogeneous interior to fill. Compartment geometry and amplitudes are designed, not fitted.
LOCAL_COMPARTMENT_PROB = 0.0
EDEMA_MARGIN = (0.2, 0.6)   # x shape radius (voxels) of dilation around the lesion: with EDEMA_KEEP this puts edema at ~1/3-2/3 of the whole lesion -- a design choice, not fitted to any annotated cohort
EDEMA_AMP = (0.3, 0.8)      # edema deviation as a fraction of the rim's, same sign
EDEMA_KEEP = 0.7            # fraction of the grown margin kept (spectral field cut) so the edema outline is irregular
_MODE_KEYS = {"Vascular-Aligned Pathology": "vascular", "Local Pathology": "local",
              "Tract-Aligned Pathology": "tract"}


def _mode_polarity(mode_name, t1_like):
    """P(bright) from POLARITY_BY_MODE for this mode and contrast, or None."""
    key = _MODE_KEYS.get(mode_name)
    if not POLARITY_BY_MODE or key not in POLARITY_BY_MODE:
        return None
    return float(POLARITY_BY_MODE[key]["t1" if t1_like else "t2"])


def _t1_like_from_gmm(S, G, mus):
    """Contrast type before rendering: compare the GMM means of the tissue
    under the WM and GM segmentation labels (modal generation label each)."""
    def modal_mu(ids):
        m = torch.isin(S, torch.tensor(ids, device=S.device, dtype=S.dtype))
        if int(m.sum()) < 100:
            return None
        return mus[int(torch.mode(G[m].long()).values)]
    wm, gm = modal_mu((1, 18)), modal_mu((2, 19))
    if wm is None or gm is None:
        return None
    return bool(wm > gm)
TISSUE_SUB_CLASSES = {"csf": (3, 20, 11, 12), "gm": (2, 19)}   # S ids used to look the G label up
TISSUE_SUB_CSF_FRAC = 0.8
# Composition lesions: with probability LESION_COMPOSITION[mode] each zone
# renders at mu = (1 - f) * mu_host + f * mu_target with the host's sigma,
# and the Rose floor rebuild and the 2*tau ceiling leave it as drawn.
# (zone, target class of TISSUE_SUB_CLASSES, f range). Wardlaw et al. 2013
# (STRIVE): lacune CSF-like on all sequences, WMH on T1 iso/hypointense but
# not as hypointense as CSF; Allen et al. RadioGraphics 2012: chronic
# encephalomalacia follows CSF; tumour necrosis is T1-hypointense / T2-hyperintense (Upadhyay & Waldman,
# Br J Radiol 2011;84:S107, PMC3473894 -- a review, not a dataset).
# train.py --lesion_comp.
LESION_COMPOSITION = {}
LESION_COMPOSITION_ZONES = {
    'vascular': [('core', 'csf', (0.7, 0.95)), ('shell', 'gm', (0.3, 1.0))],   # csf upper bound 0.95: a cavity holds CSF plus debris/gliosis, never darker than CSF
    'local':    [('core', 'csf', (0.5, 0.95)), ('shell', 'gm', (0.1, 0.5))],
    'tract':    [('whole', 'gm', (0.3, 1.0))],
}
# Rose gate against the host tissue's own noise: the core/whole zone is
# applied only if f * |mu_host - mu_target| >= COMP_MIN_K * sigma_host in
# GMM units (cf. pathology_primitives.ROSE_CRITERION_K); otherwise the
# ordinary lesion path runs. Contrast-agnostic by construction; coverage is
# P times the fraction of draws with a visible host-target gap.
COMP_MIN_K = 1.0
COMP_F_CSF = None   # (lo, hi) overriding the csf zones' f range; f > 1 overshoots the fluid level (darker than CSF on T1-like draws)
# False keeps the geometric lesion label where |dI| <= tau. train.py --target_region.
LABEL_SUPPRESS_INVISIBLE = True
LESION_FORCE_RINGS = True   # False -> the primitives' own 50% ring draw
LESION_MAX_ZONES = None     # int -> cap n_lesion_zones (1 = one GMM zone per lesion)
# The actual interior-roughness lever (found 2026-09-13 ~19:20 UTC). In the
# floor branch of _correct_lesion_visibility -- the common case, a lesion too
# subtle to clear tau -- the zone rendering is DISCARDED and the lesion is
# rebuilt as clean + room*tanh(k*blur(delta)/room), with room = 1-clean or
# clean PER VOXEL. That pushes the clean tissue's own voxel noise through a
# kinked function inside the lesion: voxel-scale roughness the zone/sigma/ring
# knobs cannot touch (all four failed). Real interiors are SMOOTHER than their
# surround. A float here blurs the clean reference used for both the base and
# room inside the lesion; outside it the image is untouched. None = old.
LESION_CLEAN_REF_SMOOTH_SIGMA = None
# One line per lesion to stdout: the patched globals as SEEN inside the
# closure, and which visibility branch ran. Diagnostic only.
VISIBILITY_DEBUG = False

# Private stream for the composition coin and f, so the shared CPU/device
# streams see the same draws with the feature on or off.
_COMPOSITION_RNG = None


def _composition_rand():
    global _COMPOSITION_RNG
    if _COMPOSITION_RNG is None:
        _COMPOSITION_RNG = torch.Generator().manual_seed(torch.initial_seed())
    return torch.rand(1, generator=_COMPOSITION_RNG).item()


def _draw_composition(mode_key):
    """[(zone, class, f)] when the coin lands, else None. Nothing is drawn at P = 0."""
    p = LESION_COMPOSITION.get(mode_key, 0.0) if mode_key else 0.0
    if p <= 0 or _composition_rand() >= p:
        return None
    zones = [(zone, cls, (COMP_F_CSF if (cls == 'csf' and COMP_F_CSF) else (lo, hi)))
             for zone, cls, (lo, hi) in LESION_COMPOSITION_ZONES[mode_key]]
    return [(zone, cls, lo + (hi - lo) * _composition_rand()) for zone, cls, (lo, hi) in zones]


def _composition_mu(mu_host, mu_target, f):
    return (1.0 - f) * mu_host + f * mu_target


def _modal_label(G, mask):
    hl, hc = torch.unique(G[mask].long(), return_counts=True)
    return int(hl[hc.argmax()])


def _composition_visible(mus, sigmas, host, target, f):
    """Rose gate: expected zone contrast against the host sigma, GMM units."""
    return float(f) * float((mus[host] - mus[target]).abs()) >= COMP_MIN_K * float(sigmas[host])


def _erode_core(mask, lo=0.15, hi=0.5, max_iter=8):
    """Erode until the core is lo..hi of the mask; None if no core fits."""
    m = mask.float()[None, None]; n = float(mask.sum()); core = None
    for _ in range(max_iter):
        e = -F.max_pool3d(-m, 3, stride=1, padding=1)
        if float(e.sum()) < lo * n:
            break
        m = e; core = m[0, 0] > 0.5
        if float(m.sum()) <= hi * n:
            break
    return core if core is not None and core.any() else None


def _dilate_iter(mask, r):
    """Binary dilation by r voxels as r passes of a 3x3x3 max-pool (cheap for the 10-30 voxel margins edema needs; a single
    (2r+1)^3 window would be O(r^3) per voxel)."""
    m = mask.float()[None, None]
    for _ in range(int(r)):
        m = F.max_pool3d(m, 3, stride=1, padding=1)
    return m[0, 0] > 0.5


def _grow_mask(mask, r):
    k = torch.ones((1, 1, 2 * r + 1, 2 * r + 1, 2 * r + 1), device=mask.device)
    return F.conv3d(mask.float()[None, None], k, padding=r)[0, 0] > 0.5


def _surround_band(lesion_mask, gap=3, dilate=3):
    """Band of tissue `gap` voxels out from the lesion; touching band if that
    is too small; None below 27 voxels."""
    inner = _grow_mask(lesion_mask, gap)
    band = _grow_mask(lesion_mask, gap + dilate) & ~inner
    if band.sum() < 27:
        band = _grow_mask(lesion_mask, dilate) & ~lesion_mask
        if band.sum() < 27:
            return None
    return band


def _host_label_sd(img, lesion_mask, seg_clean, min_vox=27):
    """sd of img over band voxels carrying the lesion's modal clean label;
    half the band sd below min_vox; None with no band. seg_clean: indices or one-hot."""
    band = _surround_band(lesion_mask)
    if band is None:
        return None
    def _idx(m):
        v = seg_clean[m]
        return v.long() if v.dim() == 1 else v.argmax(dim=-1)
    host = torch.mode(_idx(lesion_mask)).values
    same = _idx(band) == host
    if int(same.sum()) >= min_vox:
        return img[band][same].std()
    return 0.5 * img[band].std()


def _contrast_bright_prior(image, seg, label_list, prior=POLARITY_PRIOR):
    """P(lesion brighter than its surround) implied by the drawn contrast.

    Water-rich pathology reads dark where white matter is brighter than cortex
    (T1-like) and bright where cortex is brighter (T2/FLAIR-like), so the sign
    is a property of the contrast, not of the pathology alone. Drawing it
    50/50 instead breaks that link, which is what cost the bright-lesion
    evaluation set ~0.19 AUC at matched steps.

    Read off the rendered image rather than the GMM means so the bias field,
    resolution chain and normalisation are all accounted for. Falls back to
    0.5 whenever the contrast cannot be established, which keeps the caller's
    behaviour unchanged rather than guessing.
    """
    wm = [label_list.index(l) for l in (1, 18) if l in label_list]
    gm = [label_list.index(l) for l in (2, 19) if l in label_list]
    if not wm or not gm:
        return 0.5
    lab = seg if seg.dim() == 3 else seg.argmax(dim=-1)
    wm_m = torch.isin(lab, torch.tensor(wm, device=lab.device))
    gm_m = torch.isin(lab, torch.tensor(gm, device=lab.device))
    if int(wm_m.sum()) < 100 or int(gm_m.sum()) < 100:
        return 0.5
    t1_like = image[wm_m].median() > image[gm_m].median()
    return (1.0 - prior) if t1_like else prior


def _inject_pathology(S, G, device, reference_image, lesion_g_label=1, lesion_s_label=33,
                       mode_weights=(0.20, 0.15, 0.15, 0.25, 0.25), k_rose=ROSE_CRITERION_K,
                       n_lesion_zones=3, couple_prob=0.15, couple_appearance=False,
                       resolution=None, return_grid=False):
    """
    Uses PathologyRandomizer's geometric machinery (excursion-set/severity
    mask generation, ROI-anchoring, ring/core-shell geometry, mass-effect
    displacement) purely to decide WHERE pathology occurs and how it displaces
    surrounding anatomy -- NOT its own intensity-modulation machinery.
    Intensity comes from the generator's own GMM synthesis instead: G's label
    1 ('lesion') is an existing intensity class like WM/GM/CSF, and S's label
    33 ('Lesion') is an existing class in label_list_segmentation. Pathology
    then goes through the same downstream GMM synthesis, deformation, bias
    field, resolution, and noise as everything else.

    reference_image: a real image (e.g. native-space T1), needed so
    PathologyRandomizer's internal Rose-criterion visibility gate has a
    meaningful noise floor to compare against -- a constant placeholder makes
    that check numerically arbitrary (see pathology_primitives.py). The
    intensity image PathologyRandomizer itself produces is discarded either
    way; only the noise-floor measurement needs to be real.

    n_lesion_zones: number of distinct GMM intensity zones within the lesion
    region. Each zone gets its own independently-drawn (mu, sigma) from the
    GMM, using temporary G labels 251..(251+n_lesion_zones-1) -- above the
    partial-volume band (100-250) and below the 256-entry table limit, so
    they don't collide with any real generation_labels.nii values. S_path is
    still all-33 regardless of zones -- the network sees one Lesion class with
    spatially heterogeneous appearance, not n_lesion_zones separate classes.
    Zones are private to G; they never appear in S or any downstream output.

    Operates in NATIVE (pre-deformation, pre-GMM-synthesis) space.

    Returns S_path, G_path (same shape/dtype as inputs), a metadata dict,
    fold_native: (D,H,W) bool, True where the mass-effect displacement
    locally folded (non-invertible Jacobian), and (S_disp, G_disp): the
    displaced maps before the lesion label is painted in.
    """
    from pathology_primitives import PathologyRandomizer

    shape = tuple(S.shape[-3:])
    aug = PathologyRandomizer((1, 1, *shape), device=device)

    S_native = S.reshape(1, 1, *shape).float()
    ref_native = reference_image.reshape(1, 1, *shape).float()
    # couple_prob passed through so callers can isolate a single mechanism.
    # mode_weights alone only picks the PRIMARY mode -- each of the other four
    # is then independently added with probability couple_prob, so a fraction
    # 1 - (1 - couple_prob)^4 of draws carry an extra mechanism. That
    # contaminates any measurement meant to characterise one mode.
    out = aug.augment_batch(ref_native, seg_tensor=S_native, batch_size=1,
                            mode_weights=mode_weights, k_rose=k_rose,
                            couple_prob=couple_prob, force_rings=LESION_FORCE_RINGS)

    relabel_mask = (out['seg_with_novel_label'] != out['warped_seg'])[0, 0]  # (D,H,W) bool

    S_path = out['warped_seg'][0, 0].clone()
    S_path[relabel_mask] = float(lesion_s_label)  # all zones collapse to 33 in S
    S_path = S_path.round().to(S.dtype)

    G_native = G.reshape(1, 1, *shape).float()
    G_warped = F.grid_sample(G_native, out['grid'], mode='nearest', padding_mode='border', align_corners=False)[0, 0]
    G_path = G_warped.clone()

    # couple_appearance: derive internal heterogeneity from lesion SIZE rather
    # than fixing it. The GMM otherwise draws a lesion's appearance with no
    # reference to its geometry -- a 500-voxel lesion and a 200,000-voxel one
    # get the same number of intensity zones drawn the same way.
    #
    # Size-dependent heterogeneity is real biology, not a tuning choice: a
    # small lesion is a homogeneous focus, while a large mass outgrows its
    # blood supply and develops a necrotic core, an enhancing rim, and
    # surrounding edema. Thresholds are in voxels at the working resolution
    # and are order-of-magnitude anatomical claims rather than fitted values.
    if couple_appearance and relabel_mask.any():
        # Zone count from lesion RADIUS rather than fixed thresholds on volume.
        # A concentric zone needs room to exist, so the number that can fit
        # scales with radius; a small focus is homogeneous and a large mass can
        # carry core, rim and surrounding change. That much is geometry.
        #
        # Two hard bounds, both derived rather than chosen:
        #   - a zone cannot be thinner than one voxel, so radius (in voxels)
        #     caps the count
        #   - a zone cannot be observed if it is thinner than the ACQUIRED
        #     voxel, so radius/max(resolution) caps it again after the
        #     resolution simulation blurs the working grid
        #   - MAX_ZONE_LABELS is what the GMM table leaves free above the
        #     partial-volume band
        #
        # The draw between 1 and that bound is a randomization choice, not a
        # derivation: nothing measured here determines how internally
        # structured a lesion of a given size should be. Sampling the whole
        # range keeps homogeneous and heterogeneous both in support.
        n_vox = int(relabel_mask.sum())
        radius = _local_thickness_radius(relabel_mask)
        bound = radius
        if resolution is not None:
            bound = min(bound, radius / max(float(np.max(resolution)), 1e-6))
        max_zones = int(np.clip(bound, 1, MAX_ZONE_LABELS))
        n_lesion_zones = 1 + int(torch.rand(1, device=device).item() * max_zones)
        if LESION_MAX_ZONES is not None:
            n_lesion_zones = min(n_lesion_zones, int(LESION_MAX_ZONES))

    # A structure whose true width can't meaningfully support even one zone
    # boundary at the field's own floor scale gets forced to a single zone,
    # REGARDLESS of whether n_lesion_zones came from the fixed default above
    # (couple_appearance=False, i.e. every run so far) or the size-adaptive
    # computation just above (couple_appearance=True) -- this check has to
    # apply either way, since the underlying problem (a thin, winding shape
    # getting subdivided by a field whose minimum wavelength is wider than
    # the shape itself) doesn't depend on which path set n_lesion_zones.
    # Verified directly: a winding tract of true cross-sectional radius 3
    # gets the same ZONE_SCALE_FLOOR-dominated scale of 6.0 whether the OLD
    # (equivalent-sphere-from-volume) or NEW (_local_thickness_radius)
    # formula computed its size, since 0.3*radius is below the floor either
    # way -- fixing the radius estimate alone doesn't reach this case, only
    # forcing max_zones down here does.
    ZONE_SCALE_FRACTION = 0.3
    ZONE_SCALE_FLOOR = 6.0
    shape_radius = _local_thickness_radius(relabel_mask) if relabel_mask.any() else 1.0
    if n_lesion_zones > 1 and ZONE_SCALE_FRACTION * shape_radius < ZONE_SCALE_FLOOR:
        n_lesion_zones = 1

    out['metadata']['concentric'] = False
    out['metadata']['compartments'] = None
    if (LOCAL_COMPARTMENT_PROB > 0 and relabel_mask.any()
            and _MODE_KEYS.get(out['metadata'].get('primary_mode_name')) == 'local'
            and torch.rand(1, device=device).item() < LOCAL_COMPARTMENT_PROB):
        _core = None
        if out['metadata'].get('is_ring'):
            _swl = out['seg_with_novel_label'][0, 0]
            _c = (_swl == float(out['novel_label_core'])) & relabel_mask
            if float(_c.sum()) >= 0.05 * float(relabel_mask.sum()) and (relabel_mask & ~_c).any():
                _core = _c
        if _core is None:
            _core = _erode_core(relabel_mask)
        if _core is not None and (relabel_mask & ~_core).any():
            _rad = _local_thickness_radius(relabel_mask)
            _m = int(max(1, round(float(np.random.uniform(*EDEMA_MARGIN)) * _rad)))
            _grown = _dilate_iter(relabel_mask, _m) & ~relabel_mask
            _tissue = (G_warped.round() == 2) | (G_warped.round() == 3)   # healthy WM/GM only: edema does not enter CSF or leave the brain
            _edema = _grown & _tissue
            if _edema.any():
                _f = aug._universal_spectral_field(batch_size=1, max_freq=1.0 / max(ZONE_SCALE_FLOOR, ZONE_SCALE_FRACTION * _rad), force_dc=False).squeeze(0).squeeze(0)
                _cut = torch.quantile(_f[_edema], 1.0 - EDEMA_KEEP)
                _edema = _edema & (_f >= _cut)
            if _edema.any():
                S_path[_edema] = float(lesion_s_label)            # edema is lesion (whole-tumour convention)
                relabel_mask = relabel_mask | _edema              # everything downstream sees the whole lesion
                G_path[_core] = float(ZONE_G_LABEL_BASE)
                G_path[relabel_mask & ~_core & ~_edema] = float(ZONE_G_LABEL_BASE + 1)
                G_path[_edema] = float(ZONE_G_LABEL_BASE + 2)
                out['metadata']['concentric'] = True              # zone 0 is the core for every downstream consumer
                out['metadata']['compartments'] = {'core_frac': round(float(_core.sum()) / float(relabel_mask.sum()), 3),
                                                   'edema_frac': round(float(_edema.sum()) / float(relabel_mask.sum()), 3),
                                                   'margin_vox': _m}
                out['metadata']['_edema'] = _edema                # popped at the zone-mu stage
    if out['metadata']['concentric']:
        pass
    elif CONCENTRIC_ZONES and out['metadata'].get('is_ring') and relabel_mask.any():
        _swl = out['seg_with_novel_label'][0, 0]
        _core = (_swl == float(out['novel_label_core'])) & relabel_mask
        _shell = relabel_mask & ~_core
        if float(_core.sum()) >= 0.05 * float(relabel_mask.sum()) and _shell.any():
            G_path[_core] = float(ZONE_G_LABEL_BASE)
            G_path[_shell] = float(ZONE_G_LABEL_BASE + 1)
            out['metadata']['concentric'] = True
    if out['metadata']['concentric']:
        pass
    elif n_lesion_zones <= 1 or not relabel_mask.any():
        # Degenerate case: single zone, original behaviour
        G_path[relabel_mask] = float(lesion_g_label)
    else:
        # Subdivide the lesion region into n_lesion_zones distinct GMM classes
        # using a spectral random field thresholded into equal-probability
        # quantile bins -- so zones are roughly equal in size regardless of
        # the field's exact distribution, and the zone boundaries are
        # spatially coherent (correlated at the same scale as the lesion's
        # own spatial structure).
        # Temporary G labels ZONE_G_LABEL_BASE.. : above the partial-volume
        # band, below the GMM table limit, and not present in real
        # generation_labels.nii files (which use only low anatomical indices).
        # These are ONLY written to G_path here and used during GMM synthesis;
        # they never appear in S_path or any output the network sees.
        #
        # Zone spatial scale MUST track lesion size, not use a fixed
        # frequency: a field with fixed ~6-7 voxel coherence, quantile-
        # thresholded across a lesion much bigger than that, doesn't
        # produce large coherent chunks -- it produces many small
        # scattered islands per zone, since the field keeps flipping
        # quantile bins every ~6-7 voxels regardless of lesion size. That's
        # a salt-and-pepper generator for any lesion past a fairly small
        # size, not the core/rim/edge structure this is meant to model.
        #
        # "Lesion size" here means the shape's own local cross-sectional
        # width (_local_thickness_radius), not an equivalent-sphere radius
        # from total volume -- the latter conflates volume with size for
        # any elongated shape. A long, thin tract and a compact blob of
        # the same total volume got the SAME zone scale under the old
        # formula, even though the tract's true width can be several times
        # smaller -- exactly backwards, and reproducing the same
        # salt-and-pepper failure this fix already solved for compact
        # lesions, just along the tract's length instead of across a
        # blob's diameter.
        # Mirrors TEXTURE_SPATIAL_SCALE below, which already gets this
        # right for the (separate) texture field -- same pattern, applied
        # here too. ZONE_SCALE_FRACTION and ZONE_SCALE_FLOOR are the same
        # constants already used above to decide whether this branch could
        # even be reached (design choices, not measured -- same status as
        # TEXTURE_SCALE_FRACTION below). shape_radius is also already
        # computed above -- reused here rather than run twice.
        zone_spatial_scale = max(ZONE_SCALE_FLOOR, ZONE_SCALE_FRACTION * shape_radius)
        zone_field = aug._universal_spectral_field(
            batch_size=1,
            max_freq=1.0 / zone_spatial_scale,
            force_dc=False,
        ).squeeze(0).squeeze(0)  # (D,H,W), zero-mean unit-std

        # Quantile-threshold within the lesion mask only, so zones are
        # equal-area regardless of the field's spatial distribution --
        # avoids one zone dominating if the field happens to be skewed
        # in the lesion region.
        field_vals = zone_field[relabel_mask]
        quantiles = torch.quantile(
            field_vals,
            torch.linspace(0, 1, n_lesion_zones + 1, device=device),
        )
        for z in range(n_lesion_zones):
            lo, hi = quantiles[z], quantiles[z + 1]
            zone_mask = relabel_mask & (zone_field >= lo) & (zone_field < hi)
            # Last zone: use <= hi so the top quantile boundary is inclusive
            if z == n_lesion_zones - 1:
                zone_mask = relabel_mask & (zone_field >= lo)
            G_path[zone_mask] = float(ZONE_G_LABEL_BASE + z)

    G_path = G_path.round().to(G.dtype)

    fold_native = _mass_effect_jacobian_determinant(out['grid']) <= 0

    out['metadata']['tissue_sub'] = None
    if TISSUE_SUB and relabel_mask.any():
        _mode_key = {"Vascular-Aligned Pathology": "vascular", "Local Pathology": "local",
                     "Tract-Aligned Pathology": "tract"}.get(out['metadata'].get('primary_mode_name'))
        if _mode_key and torch.rand(1).item() < TISSUE_SUB.get(_mode_key, 0.0):
            _cls = "csf" if torch.rand(1).item() < TISSUE_SUB_CSF_FRAC else "gm"
            _S = S_native[0, 0]
            _src = torch.isin(_S, torch.tensor(TISSUE_SUB_CLASSES[_cls], device=_S.device, dtype=_S.dtype))
            if _src.any():
                _g = int(torch.mode(G_native[0, 0][_src].long()).values)
                _region = relabel_mask
                if _mode_key == "local" and out['metadata'].get('concentric'):
                    _region = (G_path == float(ZONE_G_LABEL_BASE))
                elif _mode_key == "local":
                    _core = _erode_core(relabel_mask)   # necrotic core, 15-50% of the lesion
                    if _core is not None:
                        _region = _core
                G_path[_region] = float(_g)
                out['metadata']['tissue_sub'] = f"{_cls}:{_g}"

    out['metadata']['composition'] = None
    out['metadata']['composition_drawn'] = None
    out['metadata']['composition_skipped'] = None
    if LESION_COMPOSITION and relabel_mask.any() and out['metadata']['tissue_sub'] is None:
        _zones = _draw_composition(_MODE_KEYS.get(out['metadata'].get('primary_mode_name')))
        if _zones:
            _S, _G = S_native[0, 0], G_native[0, 0]
            _core = None
            if len(_zones) > 1:
                _core = (G_path == float(ZONE_G_LABEL_BASE)) if out['metadata'].get('concentric') else _erode_core(relabel_mask)
                if _core is not None and not bool(_core.any()):
                    _core = None
                if _core is None:
                    _zones = [('whole',) + _zones[0][1:]]   # no core fits: the whole lesion takes the core spec
            _ed = out['metadata'].get('_edema')
            _body = relabel_mask if _ed is None else relabel_mask & ~_ed   # compartments: composition paints the tumour body, edema keeps its zone
            _masks = {'whole': _body, 'core': _core,
                      'shell': None if _core is None else _body & ~_core}
            _classes = {z[1] for z in _zones}
            _targets = {}
            for _cls in _classes:
                _src = torch.isin(_S, torch.tensor(TISSUE_SUB_CLASSES[_cls], device=_S.device, dtype=_S.dtype))
                if _src.any():
                    _targets[_cls] = int(torch.mode(_G[_src].long()).values)
            if len(_targets) == len(_classes):
                # painted into G_path at the zone-mu stage, after the Rose gate
                _rec, _paint = [], G_path[relabel_mask].clone()
                for _z, (_zone, _cls, _f) in enumerate(_zones):
                    _paint[_masks[_zone][relabel_mask]] = float(ZONE_G_LABEL_BASE + _z)
                    _rec.append((ZONE_G_LABEL_BASE + _z, _cls, _targets[_cls], float(_f)))
                out['metadata']['composition'] = ";".join(f"{z}:{c}:f={f:.2f}" for z, c, f in _zones)
                out['metadata']['composition_drawn'] = out['metadata']['composition']
                out['metadata']['composition_zones'] = _rec
                out['metadata']['composition_paint'] = (relabel_mask, _paint)

    # Displaced but un-lesioned: the healthy anatomy exactly as the
    # pathological stream sees it.
    S_disp = out['warped_seg'][0, 0].round().to(S.dtype)
    G_disp = G_warped.round().to(G.dtype)
    ret = (S_path, G_path, out['metadata'], fold_native, (S_disp, G_disp))
    # return_grid: the mass-effect sampling grid that displaced G/S, so a co-registered real image can be warped alike
    return ret + (out['grid'],) if return_grid else ret


def paired_pathology_generator(datadir,
                                mode_sampler,
                                label_list_segmentation,
                                n_neutral_labels,
                                label_list_normalization,
                                flipping=False,
                                sides='both',
                                labels_to_kill=None,
                                list_to_kill_photo=None,
                                list_to_kill_photo_inc_ventricles=None,
                                random_kill_labels=None,
                                random_kill_prob=0.0,
                                random_kill_frac=(0.5, 1.0),
                                mix_real_images=False,
                                compact_labels=False,
                                susceptibility=False,
                                susceptibility_prob=1.0,
                                couple_appearance=False,
                                couple_prob=0.15,
                                complex_resample=False,
                                gibbs_ringing_filter=True,
                                tukey_alpha=0.3,
                                generate_pathology=True,
                                minimal_io=True,
                                siz=[160, 160, 160],
                                max_rotation=15,
                                max_shear=0.2,
                                max_scaling=0.2,
                                nonlin_scale_min=0.03,
                                nonlin_scale_max=0.06,
                                nonlin_std_max=4,
                                bf_scale_min=0.02,
                                bf_scale_max=0.04,
                                bf_std_min=0.1,
                                bf_std_max=0.6,
                                bag_scale_min=0.02,
                                bag_scale_max=0.08,
                                gamma_std=0.1,
                                min_noise_std=NOISE_STD_MIN,
                                max_noise_std=NOISE_STD_MAX,
                                bag_prob=0.5,
                                deform_one_hots=True,
                                bspline_zooming=False,
                                lesion_g_label=1,
                                lesion_s_label=33,
                                mode_weights=(0.23, 0.05, 0.05, 0.335, 0.335),
                                cheap_repeat_k=3,
                                dtype=torch.float,
                                device='cpu',
                                healthy_only=False,
                                k_rose_sampler=None,
                                intra_pair_prob=0.0):
    """
    Same generative pipeline as `super_generator`, sharing its setup and
    per-case loading, except pathology is injected into the label maps G/S
    BEFORE GMM intensity synthesis and the deformation/bias-field/resolution/
    noise pipeline -- so pathological regions go through identical downstream
    augmentation as everything else. Every random draw from GMM sampling
    onward is sampled ONCE and reused for both the clean and pathological
    pass, so the only difference between the two outputs is the pathology
    itself.

    Always yields BOTH the clean and pathological case, every call -- no
    internal probability gate. Whether/how often to train on the pathological
    one is a decision for the training loop (see ptrain.py's --pathology_prob).

    Yields a dict per case:
      image_clean, seg_clean_onehot : identical to super_generator's [0], [1]
      image_path,  seg_path_onehot  : pathology-augmented counterparts,
                                       sharing all of the above randomness
      pathology_metadata            : PathologyRandomizer's per-case info
      mode                          : sampled generator mode (photo/exvivo/...)
    """
    ctx = _build_generator_context(datadir, label_list_segmentation, n_neutral_labels,
                                    label_list_normalization, list_to_kill_photo,
                                    list_to_kill_photo_inc_ventricles, siz, dtype, device, healthy_only=healthy_only)

    from pathology_primitives import PathologyRandomizer  # needed here for the contrast-retry Rose check below

    if ANOMALY_SOURCE == 'fpi' and IMAGE_SOURCE != 'real':
        raise ValueError("ANOMALY_SOURCE 'fpi' needs IMAGE_SOURCE 'real' (DISYRE's anomaly is a patch of a real scan)")
    _fpi_cache = {}   # per contrast, DISYRE's FIFO of recent foreign patches (fpi.py:68-75); filled by every expensive real draw

    def _cheap_perturb_pair(img_clean, img_path, *extra):
        # Lightweight, appearance-only perturbation for cheap-repeat passes.
        # Shares noise/gamma between clean/path streams (matching the rest of
        # this pipeline) so the only difference stays the pathology itself --
        # independent per-stream noise would add its own std-~0.07 difference
        # unrelated to pathology. Never touches labels either way.
        extra_noise_std = 0.01 + 0.04 * torch.rand(1, device=device, dtype=dtype).item()  # 1-5%
        shared_noise = torch.randn_like(img_clean)  # ONE realization, used for both streams
        gamma = 0.85 + 0.30 * torch.rand(1, device=device, dtype=dtype).item()  # [0.85, 1.15]

        def _apply(img):
            noisy = (img + shared_noise * extra_noise_std).clamp(min=1e-6, max=1.0)
            return noisy.pow(gamma)

        out = (_apply(img_clean), _apply(img_path)) + tuple(_apply(e) for e in extra)
        return out


    def _surround_stats(lesion_mask, img, gap=3, dilate=3):
        """Mean and standard deviation of the tissue immediately around a lesion.

        Detectability is local: a lesion competes with the tissue adjacent to
        it, which is what a reader compares against and what a convolutional
        receptive field sees alongside it. On real generator output, this
        band's sd runs several times larger than sensor noise, so a
        criterion built on sensor noise alone would not constrain what the
        network can actually separate.

        A dilation band rather than the generator's own core/shell structure --
        those are anatomical features that not every mode produces, whereas
        this exists for any lesion shape including multi-component ones. Its
        value stays flat across lesion size, so fragmented lesions do not
        inflate it.
        """
        # GAP before the band. Without it the reference is contaminated by the
        # lesion's own blurred edge and the correction feeds back on itself:
        # Newton boosts the lesion, the blur spreads that boosted intensity
        # into the adjacent voxels, the band's sd rises, tau rises, Newton
        # boosts further -- collapsing the available k headroom and inflating
        # more for large lesions than small ones, the signature of boundary
        # bleed rather than genuine tissue variation.
        #
        # Skipping `gap` voxels puts the band in real neighbouring tissue,
        # past the partial-volume transition.
        band = _surround_band(lesion_mask, gap, dilate)
        if band is None:
            return None, None
        return img[band].mean(), img[band].std()

    def _correct_lesion_visibility(final_clean, final_path, seg_path_onehot, seg_clean_onehot, current_k_rose,
                                    stds, II, JJ, KK, factors, M_path,
                                    newsize=None, ks_tukey=None):
        # Recomputed per call: cheap-repeat's own noise injection shifts the
        # noise floor after the label was decided, so tau can't be cached.
        # Accept either the one-hot or the argmax indices: the only thing used
        # here is which label wins, and the index form already answers that.
        if seg_path_onehot.dim() == 3:
            lesion_mask_out = seg_path_onehot == lesion_compact_idx
        else:
            lesion_mask_out = seg_path_onehot[..., lesion_compact_idx] == seg_path_onehot.max(dim=-1).values
        if not lesion_mask_out.any():
            pathology_metadata['visibility_events'] = {'floor': False, 'ceiling': False}
            pathology_metadata['suppressed_share'] = 0.0
            return final_path
        _comp = bool(pathology_metadata.get('composition'))
        if VISIBILITY_DEBUG:
            print(f"[visibility] seen: clean_ref_smooth={LESION_CLEAN_REF_SMOOTH_SIGMA} sigma_frac={LESION_SIGMA_FRACTION} force_rings={LESION_FORCE_RINGS} max_zones={LESION_MAX_ZONES} composition={pathology_metadata.get('composition')}", flush=True)
        aug_check = PathologyRandomizer((1, 1) + tuple(final_clean.shape), device=device)
        # tau = visibility floor. See ROSE_CRITERION_K in
        # pathology_primitives.py for citation.
        # tau against the SURROUND, not sensor noise. The Rose criterion is
        # about a signal competing with whatever it must be distinguished
        # from; for a segmentation network that is the neighbouring tissue.
        # Referenced to noise, requested k_rose 2.06-5.36 delivered 0.01-2.03
        # surround-sigma (correlation +0.09, 0 of 60 draws on target) and the
        # pipeline achieved a median of 0.47 -- a lesion sitting well inside
        # the normal variation of its own neighbourhood.
        _s_mean, _s_sd = _surround_stats(lesion_mask_out, final_clean)
        if _s_sd is None or _s_sd < 1e-6:
            tau = aug_check._min_visible_change(final_clean.reshape(1, 1, *final_clean.shape), k_rose=current_k_rose, fallback_noise_std_frac=0.05)
        else:
            tau = current_k_rose * _s_sd
        pathology_metadata['sd_surround'] = None if _s_sd is None else float(_s_sd)
        pathology_metadata['tau'] = float(tau)
        visible_change_signed = final_path[lesion_mask_out] - final_clean[lesion_mask_out]
        mean_visible = visible_change_signed.abs().mean()
        # Hoisted out of the floor branch below: the ceiling check further
        # down needs this regardless of whether the floor fires -- most
        # importantly in the case that never enters that branch at all (a
        # draw that's already well past tau on its own, no boost needed,
        # which is exactly the case the ceiling exists to catch).
        MAX_SEVERITY_MULTIPLE = 2.0  # design choice: same knob used as the boost's cap below
        floor_event = bool(mean_visible <= tau)
        if floor_event and VISIBILITY_DEBUG:
            print(f"[visibility] FLOOR branch (mean_visible {mean_visible:.4f} <= tau {tau:.4f}): {'kept as drawn (composition)' if _comp else 'lesion rebuilt from clean'}", flush=True)
        if floor_event and not _comp:
            # One sign per lesion, drawn against the prior the drawn contrast
            # implies rather than uniformly -- see _contrast_bright_prior.
            # Taking it from whichever direction has more headroom instead
            # ties it to local brightness, and most tissue sits below the
            # midpoint. Headroom still overrides when the drawn direction
            # cannot reach tau.
            room_pos = 1.0 - final_clean[lesion_mask_out]
            room_neg = final_clean[lesion_mask_out]
            p_bright = _contrast_bright_prior(final_clean, seg_clean_onehot,
                                              label_list_segmentation)
            _pm = _mode_polarity(pathology_metadata.get('primary_mode_name'), t1_like=(p_bright < 0.5))
            if _pm is not None and p_bright != 0.5:
                p_bright = _pm
            sign = 1.0 if torch.rand(1, device=device).item() < p_bright else -1.0
            room = room_pos if sign > 0 else room_neg
            other_room = room_neg if sign > 0 else room_pos
            if room.median() < tau and other_room.median() >= tau:
                sign = -sign
                room = other_room

            # severity (one scalar/draw, overall conspicuousness) and
            # texture (a spatial field, internal heterogeneity) are two
            # independent multiplicative sources of variation. Both use
            # unscaled sigmoid(randn()) where (0,1) is the parameter's own
            # natural range.
            # MAX_SEVERITY_MULTIPLE: design choice, not measured. Bounds the
            # PRE-chain severity target only -- Newton's method below can
            # scale it upward if the resolution chain dilutes it. room is a
            # safety cap against excessive headroom-chasing, not the target.
            MAX_SEVERITY_MULTIPLE = 2.0
            achievable_severity = min(MAX_SEVERITY_MULTIPLE, (room.median() / tau).item())
            SEVERITY_SPREAD = max(0.0, achievable_severity - 1.0)

            # Internal sub-structure size scales with lesion size, not a
            # fixed absolute. Same shape-aware radius as the zone spatial
            # scale above (_local_thickness_radius, not an equivalent-sphere
            # estimate from total volume) -- an elongated lesion's texture
            # should be scaled to its actual local width, not inflated by
            # its length.
            lesion_radius = _local_thickness_radius(lesion_mask_out)
            TEXTURE_SCALE_FRACTION = torch.sigmoid(torch.randn(1, device=device)).item()
            TEXTURE_SPATIAL_SCALE = max(1.0, TEXTURE_SCALE_FRACTION * lesion_radius)
            # Must stay <1: texture ranges over (1-SPREAD, 1+SPREAD), and
            # SPREAD>=1 lets texture go negative, flipping the correction's
            # sign at that voxel.
            TEXTURE_SPREAD = LESION_TEXTURE_SPREAD_MAX * torch.sigmoid(torch.randn(1, device=device)).item()

            severity = 1.0 + SEVERITY_SPREAD * torch.sigmoid(torch.randn(1, device=device)).item()

            # Multi-scale field via _universal_spectral_field (also used for
            # geometric placement elsewhere). max_freq ~ 1/spatial_scale;
            # the function's own cap (~half Nyquist) handles tiny lesions.
            max_freq = 1.0 / TEXTURE_SPATIAL_SCALE
            smooth = aug_check._universal_spectral_field(batch_size=1, max_freq=max_freq, force_dc=False).squeeze(0).squeeze(0)
            texture = 1.0 + TEXTURE_SPREAD * (2 * torch.sigmoid(smooth) - 1)  # field is already zero-mean, unit-std

            target_diff = sign * tau * severity * texture[lesion_mask_out]

            # Lever 3: center->rim gradient. Depth = distance to the lesion
            # boundary (EDT), normalized to [0,1] over the mask; a zero-mean
            # ramp scales the contrast up at the core and down at the rim
            # (or vice-versa by sign of the amplitude is not exposed -- core
            # is always the stronger end, matching necrotic/cavitary cores).
            if INTRA_LESION_GRADIENT > 0:
                from scipy.ndimage import distance_transform_edt
                _depth = torch.zeros_like(final_clean)
                _dm = distance_transform_edt(lesion_mask_out.detach().cpu().numpy())
                _depth[lesion_mask_out] = torch.tensor(
                    _dm[lesion_mask_out.detach().cpu().numpy()], device=device, dtype=final_clean.dtype)
                _dv = _depth[lesion_mask_out]
                _dmax = _dv.max()
                if _dmax > 0:
                    _dn = _dv / _dmax                     # 0 at rim, 1 at core
                    _ramp = 1.0 + INTRA_LESION_GRADIENT * (_dn - _dn.mean())  # zero-mean over mask
                    target_diff = target_diff * _ramp

            # Lever 4: pull a DARK lesion toward local fluid intensity. Fluid
            # is estimated as a low percentile of the clean brain intensity
            # (labeling-agnostic: CSF/fluid is the darkest tissue, and the
            # multiple label conventions here -- S ids, generation labels, raw
            # ids -- and native-vs-output grids make a CSF-label lookup
            # fragile). The fluid-relative darkening replaces the tau-relative
            # one only where it is DEEPER, so visibility is never reduced -- a
            # cavity is far past tau, the chain/Newton adds no boost, and the
            # ceiling still caps it.
            _fluid_pull = DARK_FLUID
            if CAVITY_PROB > 0 and sign < 0 and float(np.random.rand()) < CAVITY_PROB:
                _fluid_pull = 1.0                                                   # this lesion is a cavity: land at fluid level
            if _fluid_pull > 0 and sign < 0:
                _brain = final_clean[final_clean > 0]
                _fluid = (torch.quantile(_brain, 0.05) if _brain.numel() > 0
                          else final_clean[lesion_mask_out].min())
                _fluid_diff = _fluid_pull * (_fluid - final_clean[lesion_mask_out])   # <= 0
                target_diff = torch.minimum(target_diff, _fluid_diff)

            # Run through the same blur/downsample/upsample chain the rest
            # of the image went through, rather than setting sharp values
            # directly. Injected after gamma/bias-field (upstream, nothing
            # to invert) so the remaining chain is exactly linear.
            chain_input = torch.zeros_like(final_clean)
            chain_input[lesion_mask_out] = target_diff * M_path
            if stds is None:
                # k-space path. take_abs=False because this is a DIFFERENCE
                # signal: |a| - |b| != |a - b|, so taking the magnitude here
                # would break the linearity the Newton rescale below depends
                # on. No noise -- the image already carries its realization,
                # and adding more would double-count it.
                small_delta = kspace_resample(chain_input, siz, newsize, device,
                                              dtype=final_clean.dtype,
                                              complex_noise=None, tukey_window=ks_tukey,
                                              take_abs=False)
            else:
                blur_delta = gaussian_blur_3d(chain_input, stds, device, dtype=final_clean.dtype)
                small_delta = fast_3D_interp_torch(blur_delta, II, JJ, KK, 'linear', device, dtype=final_clean.dtype)
            resized_delta = myzoom_torch(small_delta, 1 / factors, device)
            # Chain(k*x) = k*Chain(x) since it's linear -- cache at k=1 and
            # rescale instead of re-running the chain per iteration.
            raw_delta_unit = resized_delta / M_path
            # Smoothed reference inside the lesion (see LESION_CLEAN_REF_SMOOTH_SIGMA).
            if LESION_CLEAN_REF_SMOOTH_SIGMA is not None:
                _s = float(LESION_CLEAN_REF_SMOOTH_SIGMA)
                _clean_ref = gaussian_blur_3d(final_clean, _s * np.ones(3), device, dtype=final_clean.dtype)
                _clean_base = torch.where(lesion_mask_out, _clean_ref, final_clean)
            else:
                _clean_ref = final_clean
                _clean_base = final_clean
            room = torch.where(raw_delta_unit >= 0, 1.0 - _clean_ref, _clean_ref) + 1e-6

            # Newton's method on tanh's exact derivative (sech^2) to solve
            # for the scale k that clears tau -- quadratically convergent.
            # Targets tau+NEWTON_EPS, not tau exactly: this saturating,
            # concave function approaches its target strictly from below and
            # never technically crosses it, so a floating-point-scale
            # epsilon is needed for the loop to terminate above tau.
            NEWTON_EPS = 1e-4
            target_visible = tau + NEWTON_EPS
            MAX_NEWTON_ITERS = 6  # compute budget; quadratic convergence rarely needs this many
            k = 1.0
            tanh_val = torch.tanh(k * raw_delta_unit / room)
            final_path = _clean_base + room * tanh_val
            # Distance from the SURROUND mean rather than from the clean image
            # at the same voxels. A lesion can differ from what was previously
            # there by 4 sigma and still sit exactly at its neighbours'
            # intensity -- invisible. This is the quantity that predicts lesion
            # Dice (+0.40) and the one Newton must drive.
            # On the fallback path (_s_mean None, surround too small to
            # measure) revert to the previous quantity rather than crashing.
            _ref = _s_mean if _s_mean is not None else final_clean[lesion_mask_out].mean()
            achieved = (final_path[lesion_mask_out].mean() - _ref).abs()
            if achieved <= tau:
                for _ in range(MAX_NEWTON_ITERS):
                    sech2 = 1.0 - tanh_val ** 2
                    f_prime = (raw_delta_unit[lesion_mask_out].abs() * sech2[lesion_mask_out]).mean() + 1e-8
                    k = k + ((target_visible - achieved) / f_prime).item()
                    tanh_val = torch.tanh(k * raw_delta_unit / room)
                    final_path = _clean_base + room * tanh_val
                    achieved = (final_path[lesion_mask_out].mean() - _ref).abs()
                    if achieved >= target_visible:
                        break
                else:
                    # Exhausted iterations: room is smaller than tau at
                    # enough voxels that the mean can't be reached. Fall
                    # through to per-voxel suppression below.
                    pass

        # --- Ceiling: symmetric to the floor above --------------------------
        # GMM zones are drawn independently per zone from the full
        # anatomical mu/sigma range (see n_lesion_zones in
        # _inject_pathology), with no relationship to k_rose or to each
        # other. Nothing upstream of this point stops that from landing far
        # past what was actually requested -- the floor above guarantees a
        # MINIMUM, but there was no maximum. Contrast beyond a couple of
        # multiples of tau doesn't buy anything: real internal
        # heterogeneity isn't drawn from an unconstrained global intensity
        # table either, so it doesn't help chi, and an appearance target
        # with no consistent relationship to k_rose makes R_syn strictly
        # harder for no benefit.
        #
        # Deliberately recomputed the same way as mean_visible above (mean
        # of PER-VOXEL absolute differences), NOT the signed
        # (final_path.mean() - ref).abs() the Newton loop above uses for
        # its own convergence check. That signed version cancels for a
        # multi-zone lesion with one zone pushed dark and another bright --
        # on a symmetric dark-zone/bright-zone case it reports ~0 and never
        # fires, while the per-voxel version correctly reports the true
        # magnitude.
        #
        # Shrinking toward final_clean is safe by construction -- it can't
        # leave [0,1], unlike the boost direction above, so (unlike the
        # tanh/Newton machinery needed there) a direct linear scale
        # suffices. Valid because the chain up to this point is linear
        # (same reasoning as the k*raw_delta_unit rescaling above): scaling
        # an already-chain-processed delta by a constant is equivalent to
        # having scaled the pre-chain input by that constant and
        # reprocessing. Reuses MAX_SEVERITY_MULTIPLE as the ceiling
        # multiple too, rather than a second magic number -- one knob for
        # "how many multiples of tau is still fine" in both directions.
        achieved_now = (final_path[lesion_mask_out] - final_clean[lesion_mask_out]).abs().mean()
        tau_ceiling = MAX_SEVERITY_MULTIPLE * tau
        ceiling_event = bool(achieved_now > tau_ceiling)
        if ceiling_event:
            if VISIBILITY_DEBUG:
                print(f"[visibility] CEILING (mean |dI| {achieved_now:.4f} > 2*tau {tau_ceiling:.4f}): {'kept as drawn (composition)' if _comp else 'rescaled to 2*tau'}", flush=True)
            if not _comp:
                s = (tau_ceiling / achieved_now).item()
                final_path = final_clean + s * (final_path - final_clean)
        pathology_metadata['visibility_events'] = {'floor': floor_event, 'ceiling': ceiling_event}

        # Per-voxel label suppression (LABEL_SUPPRESS_INVISIBLE): applied after the
        # final result is determined, whether Newton succeeded, exhausted,
        # the initial k=1 attempt already passed the mean check, or the
        # ceiling above just fired. The mean passing tau doesn't mean every
        # voxel does -- a labeled voxel where |final_path - final_clean| <=
        # tau has no individually visible signal, so claiming it as Lesion
        # is false supervision. ONLY the label reverts: the image keeps
        # whatever the GMM synthesis produced, since that signal went
        # through the full gamma/bias-field/resolution chain and is
        # genuine synthesized tissue. Reverting the image too would punch
        # flat patches of clean values into pathological surroundings -- a
        # worse artifact than the one being fixed. Modifies
        # seg_path_onehot in-place through the tensor reference --
        # propagates to the caller automatically.
        suppressed_share = 0.0
        if LABEL_SUPPRESS_INVISIBLE:
            tau_gate = tau
            if _comp:
                # gate at the host tissue's own sd, not the boundary-mixed band sd
                _sd_host = _host_label_sd(final_clean, lesion_mask_out, seg_clean_onehot)
                tau_gate = tau if _sd_host is None else 1.0 * _sd_host
            per_voxel_visible = (final_path[lesion_mask_out] - final_clean[lesion_mask_out]).abs()
            invisible = per_voxel_visible <= tau_gate
            if invisible.any():
                suppress_mask = lesion_mask_out.clone()
                suppress_mask_flat = suppress_mask[lesion_mask_out]
                suppress_mask_flat[:] = invisible
                suppress_mask[lesion_mask_out] = suppress_mask_flat
                # Restore clean anatomy label rather than zeroing -- zeroing
                # leaves all channels zero at these voxels (since _inject_pathology
                # set them to one-hot Lesion only), so argmax picks 0 (background)
                # and they render as black.
                seg_path_onehot[suppress_mask] = seg_clean_onehot[suppress_mask]
                suppressed_share = float(invisible.float().mean())
        pathology_metadata['suppressed_share'] = suppressed_share
        return final_path

    _repeat_counter = 0
    _cached = None
    _intra = None      # intra_pair_prob (2026-10-27): the previous draw's injected maps, pending a second pose
    _n_yield = 0       # draws yielded so far; a twin is only queued after an EVEN-indexed draw, so a consumer taking
                       # draws in pairs always receives the twin as the SECOND of a pair (train_downstream.register_step)

    with torch.no_grad():
        while True:
            # Dynamically pull the K threshold for this iteration
            current_k_rose = k_rose_sampler() if k_rose_sampler is not None else ROSE_CRITERION_K

            if _repeat_counter > 0:
                # Cheap-repeat pass: reuse the last expensive draw entirely,
                # skip _load_and_prepare_case/_inject_pathology/synthesis/
                # deform/bias-field/resolution/noise altogether.
                _repeat_counter -= 1
                cheap_result = dict(_cached)
                (cheap_result['image_clean'], cheap_result['image_path'],
                 cheap_result['image_clean_hr'], cheap_result['image_path_hr']) = _cheap_perturb_pair(
                    _cached['image_clean'], _cached['image_path'], _cached['image_clean_hr'], _cached['image_path_hr'])
                if FPI_VISIBILITY or ANOMALY_SOURCE != 'fpi':
                  cheap_result['image_path'] = _correct_lesion_visibility(
                    cheap_result['image_clean'], cheap_result['image_path'], cheap_result['seg_path_onehot'], cheap_result['seg_clean_onehot'], _cached['k_rose'],
                    _cached['stds'], _cached['II'], _cached['JJ'], _cached['KK'], _cached['factors'], _cached['M_path'],
                    newsize=_cached['newsize'], ks_tukey=_cached['ks_tukey'])
                cheap_result['is_cheap_repeat'] = True
                cheap_result['intra_pair'] = False
                _n_yield += 1
                yield _maybe_compact(cheap_result, compact_labels)
                continue

            intra_flag = False
            if _intra is None:
                case = _load_and_prepare_case(ctx, mode_sampler, sides, labels_to_kill,
                                               max_rotation, max_shear, max_scaling,
                                               nonlin_scale_min, nonlin_scale_max, nonlin_std_max,
                                               bag_scale_min, bag_scale_max, bag_prob,
                                               siz, dtype, device,
                                               random_kill_labels=random_kill_labels,
                                               random_kill_prob=random_kill_prob,
                                               random_kill_frac=random_kill_frac,
                                               minimal_io=minimal_io)

                G_clean, S_clean = case['G'], case['S']
                T1, T2, FLAIR = case['T1'], case['T2'], case['FLAIR']
                xx2, yy2, zz2 = case['xx2'], case['yy2'], case['zz2']
                mode, resolution, thickness = case['mode'], case['resolution'], case['thickness']

                X_real, real_name, _grid = None, None, None
                if IMAGE_SOURCE == 'real':
                    # one of the subject's real volumes, uniform over what it has on disk (an RNG draw only in this mode)
                    avail = [('t1', T1)] + [(n, v) for n, v in (('t2', T2), ('flair', FLAIR)) if v is not None]
                    real_name, X_real = avail[np.random.randint(len(avail))]
                    X_real = X_real.clamp_min(0)   # the gamma stage needs a non-negative image, as _synth's clamp guarantees

                # Reject no-op draws before the expensive shared downstream
                # pipeline runs twice -- ~19% of draws produce S_path identical
                # to S_clean (geometry that fired but produced zero voxel
                # change), which would otherwise pay full price for a
                # pathological pass that's bit-for-bit the same as the clean one.
                if not generate_pathology:
                    # Baseline runs (--do_pathology off) never read image_path, but
                    # the pathological stream was being built anyway: injection with
                    # up to 8 retries, a second GMM synthesis, a second deformation,
                    # a second gamma/bias/blur/resample chain, and the Newton
                    # visibility correction -- all discarded by the training loop.
                    # Alias to the clean maps so the rest of the code runs
                    # unchanged; the duplicated work is skipped at each site below.
                    S_path, G_path = S_clean, G_clean
                    pathology_metadata = {'mode_name': 'none (generate_pathology=False)'}
                    fold_native = torch.zeros_like(S_clean, dtype=torch.bool)
                elif ANOMALY_SOURCE == 'fpi':
                    # DISYRE-kind anomaly on the real scan (2026-10-20): no label-space prior, no mass effect; the thresholded
                    # shape is the lesion label, so omega_v / ROI / loss see it exactly as a prior lesion.
                    _brain = (S_clean > 0) & ~torch.isin(S_clean, torch.arange(34, 47, device=S_clean.device, dtype=S_clean.dtype))
                    for _retry in range(8):
                        HR_fpi, _shape_fpi, _fmeta = _fpi_anomaly(X_real, _brain, _fpi_cache, real_name, case['stem'], device, dtype)
                        S_path = S_clean.clone()
                        S_path[_shape_fpi >= 0.5] = lesion_s_label   # the iso-level the reference re-binarizes at (mask_generation.py:43)
                        if not torch.equal(S_path, S_clean):
                            break
                    G_path = G_clean
                    pathology_metadata = {'mode_name': 'fpi', 'primary_mode_name': None, 'has_mass_effect': False, **_fmeta}
                    fold_native = torch.zeros_like(S_clean, dtype=torch.bool)
                else:
                  max_inject_retries = 8
                  for _retry in range(max_inject_retries):
                    _ret = _inject_pathology(
                        S_clean, G_clean, device, T1 if X_real is None else X_real,   # Rose noise floor on the drawn contrast
                        lesion_g_label=lesion_g_label, lesion_s_label=lesion_s_label,
                        mode_weights=mode_weights, k_rose=current_k_rose,
                        couple_prob=couple_prob, couple_appearance=couple_appearance,
                        resolution=resolution, return_grid=(IMAGE_SOURCE == 'real'))
                    S_path, G_path, pathology_metadata, fold_native, (S_disp, G_disp) = _ret[:5]
                    _grid = _ret[5] if len(_ret) > 5 else None
                    # With a displaced clean stream a lesion exists iff the map
                    # differs from the DISPLACED healthy one; otherwise keep the
                    # original check so default runs are unchanged.
                    if not torch.equal(S_path, S_disp if DISPLACE_CLEAN else S_clean):
                        break
                  if DISPLACE_CLEAN:
                    S_clean, G_clean = S_disp, G_disp
                # else: exhausted retries (astronomically rare) -- proceed with
                # the last attempt rather than hang.

            else:
                # Intra-subject pair (2026-10-27, --intra_prob): the previous draw's injected label maps (same anatomy, same
                # lesion, mass effect included) under a FRESH affine + nonlinear pose; contrast, gamma, bias field, resolution
                # and noise are redrawn below as for any draw. Mode / resolution / thickness stay the subject's. The loss is
                # untouched: the training loop only puts the pathological rendering on BOTH sides of such a pair.
                case = dict(_intra['case'])
                case['xx2'], case['yy2'], case['zz2'] = _intra_grid(ctx, case['G'].shape, case['mode'], case['resolution'], siz,
                                                                     max_rotation, max_shear, max_scaling,
                                                                     nonlin_scale_min, nonlin_scale_max, nonlin_std_max, dtype, device)
                S_clean, G_clean, S_path, G_path, pathology_metadata, fold_native = _intra['maps']
                T1, T2, FLAIR = case['T1'], case['T2'], case['FLAIR']
                xx2, yy2, zz2 = case['xx2'], case['yy2'], case['zz2']
                mode, resolution, thickness = case['mode'], case['resolution'], case['thickness']
                X_real, real_name, _grid = None, None, None
                intra_flag, _intra = True, None

            # --- everything from here shares randomness between both passes ---
            # Visibility is checked on the FINAL synthesized image, not just
            # the label decision -- contrast is randomized independently of
            # geometry, so a Rose-passing LABEL can still land on a
            # randomly-drawn contrast where Lesion's mu sits close to its
            # neighbor's. Only Lesion voxels are checked; healthy anatomy is
            # left to blend untouched. See _correct_lesion_visibility below.
            lesion_compact_idx = label_list_segmentation.index(lesion_s_label)
            mus = GMM_MEAN_MIN + (GMM_MEAN_MAX - GMM_MEAN_MIN) * torch.rand(GMM_TABLE_SIZE, dtype=dtype, device=device)
            sigmas = 5 + 20 * torch.rand(256, dtype=dtype, device=device)
            if mode == 'photo':
                mus[0] = 0
                sigmas[0] = 0
            elif np.random.rand() < 0.5:
                mus[0] = 0
            if DARK_CSF_PROB > 0 and IMAGE_SOURCE != 'real' and np.random.rand() < DARK_CSF_PROB:
                mus = _dark_csf(mus)
            if GMM_SIGMA_REL is not None:
                _lo, _hi = GMM_SIGMA_REL
                sigmas = mus * (_lo + (_hi - _lo) * torch.rand(mus.shape[0], dtype=dtype, device=device))
                if mode == 'photo': sigmas[0] = 0

            v_pv = 0.02 * torch.arange(50).to(device)
            mus[100:150] = mus[1] * (1 - v_pv) + mus[2] * v_pv
            mus[150:200] = mus[2] * (1 - v_pv) + mus[3] * v_pv
            mus[200:250] = mus[3] * (1 - v_pv) + mus[4] * v_pv
            mus[250] = mus[4]
            sigmas[100:150] = torch.sqrt(sigmas[1]**2 * (1 - v_pv) + sigmas[2]**2 * v_pv)
            sigmas[150:200] = torch.sqrt(sigmas[2]**2 * (1 - v_pv) + sigmas[3]**2 * v_pv)
            sigmas[200:250] = torch.sqrt(sigmas[3]**2 * (1 - v_pv) + sigmas[4]**2 * v_pv)
            sigmas[250] = sigmas[4]

            # --- Zone sign consistency ------------------------------------------
            # Each zone label (ZONE_G_LABEL_BASE..) independently draws its own mu
            # from the full anatomical range above, with no relationship to the
            # others -- so for a multi-zone lesion, roughly half the zones land
            # brighter than the surrounding tissue and half land darker, purely
            # by chance. The magnitude ceiling in _correct_lesion_visibility
            # bounds each zone's deviation FROM THE REFERENCE, but says nothing
            # about the deviation BETWEEN TWO ADJACENT ZONES: a zone at +2*tau
            # sitting next to one at -2*tau each individually passes that check
            # while presenting a 4*tau jump at the boundary between them --
            # which is what actually reads as unnatural in a rendered image, not
            # the zone-vs-background contrast the ceiling measures. Pick ONE
            # sign per lesion and re-anchor every zone's mu to that side of the
            # reference tissue's own mu, preserving each zone's independently-
            # drawn MAGNITUDE (severity/heterogeneity is still real, and real
            # lesions do vary in degree) but not its sign. This bounds the
            # worst-case zone-to-zone jump at whatever the ceiling already
            # allows for a single zone, instead of double that.
            zone_mask_all = G_path >= ZONE_G_LABEL_BASE
            if zone_mask_all.any():
                zone_labels_in_use = torch.unique(G_path[zone_mask_all]).long().tolist()
                # Reference: the clean tissue these zones most likely
                # replaced. G_clean is pre-warp, so this is only
                # approximate under significant geometric displacement --
                # same caveat as underlying_seg_vol in live_diagnostics.py,
                # and for the same reason: it's a reference pick, not a
                # precision measurement. Computed once here regardless of
                # zone count -- the sigma cap below applies even to a
                # single zone, unlike the sign fix which only matters
                # once there's more than one.
                under_labels, counts = torch.unique(G_clean[zone_mask_all].long(), return_counts=True)
                reference_g_label = under_labels[counts.argmax()].item()
                reference_mu = mus[reference_g_label]
                reference_sigma = sigmas[reference_g_label]

                if len(zone_labels_in_use) > 1:
                    _p = 0.5
                    if POLARITY_BY_MODE:
                        _t1 = _t1_like_from_gmm(S_clean, G_clean, mus)
                        _pm = _mode_polarity(pathology_metadata.get('primary_mode_name'), _t1) if _t1 is not None else None
                        _p = _pm if _pm is not None else 0.5
                    sign = 1.0 if torch.rand(1, device=device).item() < _p else -1.0
                    _edema_zl = ZONE_G_LABEL_BASE + 2 if pathology_metadata.get('compartments') else None
                    _rim_dev = None
                    for zl in sorted(zone_labels_in_use):
                        if zl == _edema_zl:
                            continue
                        if pathology_metadata.get('concentric'):
                            # core and shell choose their sides independently
                            sign = 1.0 if torch.rand(1, device=device).item() < _p else -1.0
                        magnitude = (mus[zl] - reference_mu).abs()
                        mus[zl] = reference_mu + sign * magnitude
                        if zl == ZONE_G_LABEL_BASE + 1:
                            _rim_dev = mus[zl] - reference_mu
                    if _edema_zl is not None and _edema_zl in zone_labels_in_use:
                        # edema: the rim's sign at EDEMA_AMP of its deviation (milder than the tumour on every contrast)
                        _base_dev = _rim_dev if _rim_dev is not None else mus[ZONE_G_LABEL_BASE] - reference_mu
                        mus[_edema_zl] = reference_mu + float(np.random.uniform(*EDEMA_AMP)) * _base_dev

                # --- Zone sigma cap ---------------------------------------------
                # Same class of problem as mu above, but for per-voxel SCATTER
                # rather than the zone's mean: sigmas[zl] is drawn independently
                # from the same global sigmas = 5 + 20*rand(256) table used for
                # every anatomical class, with no relationship to k_rose or to
                # the reference tissue's own sigma. The mu fix above bounds each
                # zone's MEAN and keeps multiple zones on one side of the
                # reference, but a zone with a perfectly well-behaved mean can
                # still throw individual voxels far from it if its independently
                # drawn sigma happens to land near the top of the range --
                # mus[G] + sigmas[G]*randn() scatters proportionally to sigma
                # regardless of how well-controlled mu is. That's a per-voxel
                # extremity the aggregate mean-based ceiling in
                # _correct_lesion_visibility structurally cannot see, since it
                # only measures the mean of |diff| across the whole zone.
                # Applies even to a single zone (len==1), unlike the sign fix,
                # since this isn't a zone-to-zone problem.
                SIGMA_CAP_MULTIPLE = LESION_SIGMA_CAP_MULTIPLE
                for zl in zone_labels_in_use:
                    if LESION_SIGMA_FRACTION is not None:
                        sigmas[zl] = LESION_SIGMA_FRACTION * reference_sigma
                    else:
                        sigmas[zl] = torch.clamp(sigmas[zl], max=SIGMA_CAP_MULTIPLE * reference_sigma)

            # Composition: the ordinary zone stage above ran on the ordinary
            # labels. Rose gate on the core/whole zone, then paint the zones
            # and set mu between the host and target means, host sigma.
            pathology_metadata.pop('_edema', None)
            _paint = pathology_metadata.pop('composition_paint', None)
            if _paint is not None:
                _pm_mask, _pm_vals = _paint
                _cz = pathology_metadata['composition_zones']
                _zm0 = torch.zeros_like(_pm_mask)
                _zm0[_pm_mask] = _pm_vals == _cz[0][0]
                _host0 = _modal_label(G_clean, _zm0)
                if _composition_visible(mus, sigmas, _host0, _cz[0][2], _cz[0][3]):
                    G_path[_pm_mask] = _pm_vals
                    for zl, _cls, _tg, _f in _cz:
                        _zm = G_path == zl
                        if not bool(_zm.any()):
                            continue
                        _host = _modal_label(G_clean, _zm)
                        mus[zl] = _composition_mu(mus[_host], mus[_tg], _f)
                        sigmas[zl] = sigmas[_host]
                else:
                    pathology_metadata['composition'] = None
                    del pathology_metadata['composition_zones']
                    pathology_metadata['composition_skipped'] = 'below_host_noise'

            # SAME per-voxel synthesis noise for both -- voxels where G didn't
            # change get numerically identical synthesized intensity
            syn_noise = torch.randn(G_clean.shape, dtype=dtype, device=device)
            syn_noise = _skew_noise(syn_noise, G_clean, mus, GMM_SKEW)   # healthy classes of the clean map; same draw on both renderings

            # k-means sub-tissue means (see KSEG_PROB): computed on the undisplaced native crop, where G and T1 align
            _kshift = None
            if KSEG_PROB > 0 and np.random.rand() < KSEG_PROB:
                _kshift, _kmeta = _kseg_shift(case['G'], T1, mus, device, dtype, case=case, sigmas=sigmas)
                pathology_metadata['kseg'] = _kmeta

            def _synth(G):
                SYN = mus[G] + sigmas[G] * syn_noise
                if _kshift is not None:
                    SYN = SYN + _kshift      # same shift on both renderings: path - clean is unchanged
                SYN[SYN < 0] = 0
                SYN = SYN / mus[2]
                return gaussian_blur_3d(SYN, 0.5 * np.ones(3), device, dtype=dtype)

            SYNblur_clean = _synth(G_clean)
            SYNblur_path = SYNblur_clean if (not generate_pathology or ANOMALY_SOURCE == 'fpi') else _synth(G_path)
            if HEALTHY_HETERO > 0 and IMAGE_SOURCE != 'real':
                # same field on both renderings: the lesion target is the difference and stays exact
                _F_het = _healthy_hetero_field(G_clean, device, dtype)
                SYNblur_clean = SYNblur_clean * _F_het
                SYNblur_path = SYNblur_path * _F_het
            if CSF_MIX_PROB > 0 and IMAGE_SOURCE != 'real' and np.random.rand() < CSF_MIX_PROB:
                # second CSF component (2026-10-07): same field on both renderings, the deviation target stays exact
                _F_csf = _csf_mix_field(G_clean, mus, device, dtype)
                SYNblur_clean = SYNblur_clean * _F_csf
                SYNblur_path = SYNblur_path * _F_csf

            if IMAGE_SOURCE == 'real':
                # Data-kind arms (2026-10-20): the clean image is the real scan; the pathological one is the same scan, warped
                # by the field that displaced G/S (nearest for labels at _inject_pathology, bilinear for the image as the
                # primitives warp their own), plus the anomaly. The GMM renderings only feed the lesion paint from here on.
                X_w = X_real if _grid is None else F.grid_sample(X_real[None, None].float(), _grid, mode='bilinear',
                                                                 padding_mode='border', align_corners=False)[0, 0].to(dtype)
                HR_clean = X_w if DISPLACE_CLEAN else X_real
                if not generate_pathology:
                    HR_path = HR_clean
                elif ANOMALY_SOURCE == 'fpi':
                    HR_path = HR_fpi
                else:
                    HR_path, _pmeta = _real_lesion_paint(X_w, SYNblur_path, SYNblur_clean, X_real, case['G'],
                                                         S_path == lesion_s_label, mus, sigmas, device, dtype)
                    pathology_metadata['real_paint'] = _pmeta
                pathology_metadata['real_contrast'] = real_name
            # See mix_real_images note in super_generator -- applied
            # identically to both the clean and pathological stream.
            elif mix_real_images and np.random.rand() < 0.5:
                v = torch.rand(4)
                v[2] = 0 if T2 is None else v[2]
                v[3] = 0 if FLAIR is None else v[3]
                v /= torch.sum(v)
                HR_clean = v[0] * SYNblur_clean + v[1] * T1
                HR_path = v[0] * SYNblur_path + v[1] * T1
                if T2 is not None:
                    HR_clean = HR_clean + v[2] * T2
                    HR_path = HR_path + v[2] * T2
                if FLAIR is not None:
                    HR_clean = HR_clean + v[3] * FLAIR
                    HR_path = HR_path + v[3] * FLAIR
            else:
                HR_clean = SYNblur_clean
                HR_path = SYNblur_path

            # deform BOTH with the identical grid/weights
            HRdef_clean, lin_weights = fast_3D_interp_torch(HR_clean, xx2, yy2, zz2, 'linear', device, return_linear_weights=True, dtype=dtype)
            HRdef_path = HRdef_clean if not generate_pathology else fast_3D_interp_torch_linear_given_weights(HR_path, lin_weights, 'linear', device, dtype=dtype)

            # Same shared deformation applied to the native-space fold
            # indicator, so it lands aligned with everything else in output
            # space. Thresholded back to boolean after linear interpolation.
            fold_def = fast_3D_interp_torch_linear_given_weights(
                fold_native.float().unsqueeze(-1), lin_weights, 'linear', device, dtype=dtype) > 0.5

            if deform_one_hots:
                if compact_labels:
                    # Consumer only wants indices, so skip the one-hot entirely.
                    SdefIdx_clean = deform_labels_argmax(
                        ctx.lut[S_clean], lin_weights, ctx.onehotmatrix.shape[0], device)
                    SdefOneHot_clean = None
                else:
                    Sonehot_clean = ctx.onehotmatrix[ctx.lut[S_clean]]
                    SdefOneHot_clean = fast_3D_interp_torch_linear_given_weights(Sonehot_clean, lin_weights, 'linear', device, dtype=dtype)
                    SdefIdx_clean = None
                if not compact_labels:
                    # Background channel deformed with default_value=1.0 so
                    # voxels warped in from outside the volume read as
                    # background rather than as all-zero (which argmax would
                    # resolve arbitrarily). deform_labels_argmax already
                    # initialises its output to 0 = compact index of background
                    # and only writes the in-bounds voxels, so the compact path
                    # gets this for free.
                    SdefOneHot_clean[..., 0] = fast_3D_interp_torch_linear_given_weights(Sonehot_clean[..., 0], lin_weights, 'linear', device, default_value=1.0, dtype=dtype)

                if compact_labels:
                    SdefIdx_path = (SdefIdx_clean if not generate_pathology
                                    else deform_labels_argmax(
                                        ctx.lut[S_path], lin_weights,
                                        ctx.onehotmatrix.shape[0], device))
                    SdefOneHot_path = None
                else:
                    Sonehot_path = ctx.onehotmatrix[ctx.lut[S_path]]
                    SdefOneHot_path = fast_3D_interp_torch_linear_given_weights(Sonehot_path, lin_weights, 'linear', device, dtype=dtype)
                    SdefIdx_path = None
                if not compact_labels:
                    SdefOneHot_path[..., 0] = fast_3D_interp_torch_linear_given_weights(Sonehot_path[..., 0], lin_weights, 'linear', device, default_value=1.0, dtype=dtype)
            else:
                Sdef_clean = fast_3D_interp_torch(S_clean, xx2, yy2, zz2, 'nearest', device)
                SdefOneHot_clean = ctx.onehotmatrix[ctx.lut[Sdef_clean.long()]]
                Sdef_path = fast_3D_interp_torch(S_path, xx2, yy2, zz2, 'nearest', device)
                SdefOneHot_path = ctx.onehotmatrix[ctx.lut[Sdef_path.long()]]

            # Gamma transform (SAME gamma for both)
            gamma = torch.tensor(np.exp(gamma_std * np.random.randn(1)[0]), dtype=float, device=device)
            HRgamma_clean = 3.0 * (HRdef_clean / 3.0) ** gamma
            HRgamma_path = 3.0 * (HRdef_path / 3.0) ** gamma

            # Bias field (SAME field for both)
            bf_scale = bf_scale_min + np.random.rand(1) * (bf_scale_max - bf_scale_min)
            siz_BF_small = np.round(bf_scale * np.array(siz)).astype(int).tolist()
            if mode == 'photo':
                siz_BF_small[1] = np.round(siz[1] / resolution[1]).astype(int)
            BFsmall = torch.tensor(bf_std_min + (bf_std_max - bf_std_min) * np.random.rand(1), dtype=dtype, device=device) * torch.randn(siz_BF_small, dtype=dtype, device=device)
            BFlog = myzoom_torch(BFsmall, np.array(siz) / siz_BF_small, device)
            BF = torch.exp(BFlog)
            HRbf_clean = HRgamma_clean * BF
            HRbf_path = HRgamma_path * BF

            # Susceptibility dropout, applied here: after synthesis and the
            # bias field, before the resolution simulation. That is where it
            # belongs physically -- dephasing happens during acquisition, so it
            # should be blurred and downsampled along with everything else
            # rather than painted on afterwards.
            #
            # Only the PATHOLOGICAL stream gets it: the effect is caused by the
            # lesion's own susceptibility, so the clean image has no source.
            # Appearance here depends on lesion geometry rather than being
            # drawn independently of it: the dropout field IS the lesion mask
            # convolved with the dipole kernel. Fires on every pathological
            # draw when enabled -- a lesion with susceptibility always has the
            # effect; whether a given pathology has it at all is the flag's
            # decision, not a per-draw one.
            if (susceptibility and generate_pathology
                    and np.random.rand() < susceptibility_prob):
                lesion_here = (S_path == lesion_s_label)
                HRbf_path = apply_susceptibility(HRbf_path, lesion_here, device, dtype=dtype)
                pathology_metadata['susceptibility'] = True

            # Model Resolution (SAME stds for both)
            newsize = (np.array(siz) / resolution).astype(int)
            factors = np.array(newsize) / np.array(siz)

            # SAME acquisition-noise magnitude for both -- scanner noise is not
            # tissue-dependent, so sharing it is more physically apt than
            # drawing independently.
            noise_std = torch.tensor(min_noise_std + (max_noise_std - min_noise_std) * np.random.rand(1), dtype=dtype, device=device)

            if complex_resample:
                # Crop k-space instead of blurring in image space. Truncating
                # high frequencies is what a low-resolution acquisition
                # physically does, and it produces real Gibbs ringing rather
                # than the banding a blur-then-linear-interp chain leaves.
                stds = None
                II = JJ = KK = None
                k_std = kspace_noise_std(noise_std, siz, newsize)
                nx, ny, nz = int(newsize[0]), int(newsize[1]), int(newsize[2])
                # ONE noise realization and ONE window, shared by both streams
                shared_k_noise = torch.complex(
                    k_std * torch.randn((nx, ny, nz), dtype=dtype, device=device),
                    k_std * torch.randn((nx, ny, nz), dtype=dtype, device=device))
                tukey = (make_tukey_window_3d(nx, ny, nz, alpha=tukey_alpha, device=device, dtype=dtype)
                         if gibbs_ringing_filter else None)

                smallnoisy_clean = kspace_resample(HRbf_clean, siz, newsize, device, dtype=dtype,
                                                   complex_noise=shared_k_noise, tukey_window=tukey)
                smallnoisy_path = kspace_resample(HRbf_path, siz, newsize, device, dtype=dtype,
                                                  complex_noise=shared_k_noise, tukey_window=tukey)
            else:
                shared_k_noise = tukey = None
                stds = (0.85 + 0.3 * np.random.rand()) * np.log(5) / np.pi * thickness
                stds[thickness <= 1.0] = 0.0
                HRblur_clean = gaussian_blur_3d(HRbf_clean, stds, device, dtype=dtype)
                HRblur_path = gaussian_blur_3d(HRbf_path, stds, device, dtype=dtype)

                delta = (1.0 - factors) / (2.0 * factors)
                vx = np.arange(delta[0], delta[0] + newsize[0] / factors[0], 1 / factors[0])[:newsize[0]]
                vy = np.arange(delta[1], delta[1] + newsize[1] / factors[1], 1 / factors[1])[:newsize[1]]
                vz = np.arange(delta[2], delta[2] + newsize[2] / factors[2], 1 / factors[2])[:newsize[2]]
                II, JJ, KK = np.meshgrid(vx, vy, vz, sparse=False, indexing='ij')
                II = torch.tensor(II, dtype=dtype, device=device)
                JJ = torch.tensor(JJ, dtype=dtype, device=device)
                KK = torch.tensor(KK, dtype=dtype, device=device)

                small_clean = fast_3D_interp_torch(HRblur_clean, II, JJ, KK, 'linear', device, dtype=dtype)
                small_path = fast_3D_interp_torch(HRblur_path, II, JJ, KK, 'linear', device, dtype=dtype)

                final_noise = torch.randn(small_clean.shape, dtype=dtype, device=device)
                smallnoisy_clean = (small_clean + noise_std * final_noise).clamp(min=0)
                smallnoisy_path = (small_path + noise_std * final_noise).clamp(min=0)

            if bspline_zooming:
                resized_clean = interpol.resize(smallnoisy_clean, shape=siz, anchor='edge', interpolation=3, bound='dct2', prefilter=True)
                resized_path = interpol.resize(smallnoisy_path, shape=siz, anchor='edge', interpolation=3, bound='dct2', prefilter=True)
            else:
                resized_clean = myzoom_torch(smallnoisy_clean, 1 / factors, device)
                resized_path = myzoom_torch(smallnoisy_path, 1 / factors, device)

            M_clean = torch.max(resized_clean)
            M_path = torch.max(resized_path)
            final_clean = resized_clean / M_clean
            final_path = resized_path / M_path  # each normalized by its own max
            # Pre-degradation images (after contrast, gamma and bias field;
            # before blur / resampling / noise), on the degraded images'
            # scale: the super-resolution target.
            hr_clean = HRbf_clean / M_clean
            hr_path = HRbf_path / M_path

            # Called before the flip so the delta-chain blur/resample
            # operates in the same coordinate system stds/II/JJ/KK/factors
            # were computed in.
            if generate_pathology and (FPI_VISIBILITY or ANOMALY_SOURCE != 'fpi'):
              final_path = _correct_lesion_visibility(final_clean, final_path,
                                                      SdefIdx_path if SdefOneHot_path is None else SdefOneHot_path,
                                                      SdefIdx_clean if SdefOneHot_clean is None else SdefOneHot_clean,
                                                      current_k_rose,
                                                      stds, II, JJ, KK, factors, M_path,
                                                      newsize=newsize, ks_tukey=tukey)

            # SAME flip decision for both
            if flipping and (np.random.rand() < 0.5):
                final_clean = torch.flip(final_clean, [0])
                final_path = torch.flip(final_path, [0])
                hr_clean = torch.flip(hr_clean, [0])
                hr_path = torch.flip(hr_path, [0])
                fold_def = torch.flip(fold_def, [0])
                if compact_labels:
                    # ctx.vflip permutes the CHANNEL axis to swap left/right
                    # labels. On index form the same permutation becomes a
                    # lookup applied to the values: new[i] = position of i in
                    # vflip, which is vflip's inverse. Build it once.
                    vf = torch.as_tensor(ctx.vflip, dtype=torch.long, device=device)
                    inv = torch.empty_like(vf)
                    inv[vf] = torch.arange(vf.numel(), device=device)
                    SdefIdx_clean = inv[torch.flip(SdefIdx_clean, [0])]
                    SdefIdx_path = inv[torch.flip(SdefIdx_path, [0])]
                else:
                    SdefOneHot_clean = torch.flip(SdefOneHot_clean, [0])[:, :, :, ctx.vflip]
                    SdefOneHot_path = torch.flip(SdefOneHot_path, [0])[:, :, :, ctx.vflip]

            # compact_labels: yield argmax indices instead of the 56-channel
            # one-hot. Every consumer argmaxes immediately, and at 160^3 the
            # one-hot is 0.92 GB per stream against 0.033 GB for indices -- 28x.
            # That matters when cases are held in a prefetch queue: depth 2 was
            # keeping 3.74 GB of GPU memory alive, and cuFFT failed to allocate
            # its workspace inside the generator (surfacing as the misleading
            # CUFFT_INVALID_SIZE).
            #
            # Deliberately DIFFERENT keys, not the same ones holding different
            # data: anything that genuinely needs the soft one-hot then raises
            # KeyError rather than silently argmaxing an index tensor.
            result = {
                'image_clean': final_clean,
                'seg_clean_onehot': SdefIdx_clean if SdefOneHot_clean is None else SdefOneHot_clean,
                'image_path': final_path,
                'seg_path_onehot': SdefIdx_path if SdefOneHot_path is None else SdefOneHot_path,
                'image_clean_hr': hr_clean,
                'image_path_hr': hr_path,
                'pathology_metadata': pathology_metadata,
                'fold_mask': fold_def,  # (D,H,W) bool, output space -- True where the mass-effect
                                          # warp locally folded (see _mass_effect_jacobian_determinant)
                'mode': mode,
                'is_cheap_repeat': False,
                'intra_pair': intra_flag,
                'k_rose': current_k_rose,  # Save the applied threshold for the visualizer/caching
                # Cached for the cheap-repeat branch's own
                # _correct_lesion_visibility call -- this case's resolution-
                # degradation character, shared across its cheap-repeats.
                'stds': stds, 'II': II, 'JJ': JJ, 'KK': KK, 'factors': factors, 'M_path': M_path,
                'newsize': newsize, 'ks_tukey': tukey,
            }
            _cached = result
            if intra_pair_prob > 0 and not intra_flag and _n_yield % 2 == 0 and IMAGE_SOURCE != 'real' \
                    and ANOMALY_SOURCE != 'fpi' and np.random.rand() < intra_pair_prob:
                _intra = {'case': {k: v for k, v in case.items() if k not in ('xx2', 'yy2', 'zz2')},
                          'maps': (S_clean, G_clean, S_path, G_path, pathology_metadata, fold_native)}
            _n_yield += 1
            _repeat_counter = cheap_repeat_k - 1  # this yield is the 1st of cheap_repeat_k total
            yield _maybe_compact(result, compact_labels)


#############




#######################
# Auxiliary functions #
#######################


def make_tukey_window_1d(length, alpha=0.5, device='cpu', dtype=torch.float32):
    """1D Tukey (tapered cosine) window. alpha=0 rectangular, alpha=1 Hann."""
    if alpha <= 0:
        return torch.ones(length, device=device, dtype=dtype)
    elif alpha >= 1:
        return torch.hann_window(length, device=device, dtype=dtype)

    window = torch.ones(length, device=device, dtype=dtype)
    taper_len = int(alpha * (length - 1) / 2)

    if taper_len > 0:
        n = torch.arange(taper_len, device=device, dtype=dtype)
        taper = 0.5 * (1 + torch.cos(torch.pi * (2 * n / (alpha * (length - 1)) - 1)))
        window[:taper_len] = taper
        window[-taper_len:] = torch.flip(taper, dims=[0])

    return window


def make_tukey_window_3d(nx, ny, nz, alpha=0.5, device='cpu', dtype=torch.float32):
    """3D Tukey window as the outer product of three 1D windows."""
    wx = make_tukey_window_1d(nx, alpha=alpha, device=device, dtype=dtype)
    wy = make_tukey_window_1d(ny, alpha=alpha, device=device, dtype=dtype)
    wz = make_tukey_window_1d(nz, alpha=alpha, device=device, dtype=dtype)
    return wx[:, None, None] * wy[None, :, None] * wz[None, None, :]


def kspace_resample(img, siz, newsize, device, dtype=torch.float,
                    complex_noise=None, tukey_window=None, take_abs=True):
    """Resample by cropping k-space -- what a low-resolution acquisition
    physically does, rather than blurring then interpolating in image space.

    Truncating high frequencies produces genuine Gibbs ringing and correct
    partial-volume behaviour. The Tukey window apodizes that ringing the way
    a scanner's reconstruction filter would.

    complex_noise and tukey_window are passed IN rather than drawn here so the
    clean and pathological streams share one realization -- the only
    difference between the two outputs must be the pathology itself.

    take_abs=False returns the real part instead of the magnitude. The
    magnitude is correct for an image (it is what makes the noise Rician), but
    wrong for a difference signal: |a| - |b| is not |a - b|, so the visibility
    correction pushes its delta through with take_abs=False and stays linear.
    """
    cdtype = torch.complex64 if dtype in (torch.float32, torch.float) else torch.complex128

    kspace = torch.fft.fftshift(torch.fft.fftn(img))

    nx, ny, nz = int(newsize[0]), int(newsize[1]), int(newsize[2])
    kspace_new = torch.zeros((nx, ny, nz), dtype=cdtype, device=device)

    cx, cy, cz = siz[0] // 2, siz[1] // 2, siz[2] // 2
    cnx, cny, cnz = nx // 2, ny // 2, nz // 2
    min_x, min_y, min_z = min(siz[0], nx), min(siz[1], ny), min(siz[2], nz)

    sx, ex = cx - min_x // 2, cx - min_x // 2 + min_x
    sy, ey = cy - min_y // 2, cy - min_y // 2 + min_y
    sz, ez = cz - min_z // 2, cz - min_z // 2 + min_z
    snx, enx = cnx - min_x // 2, cnx - min_x // 2 + min_x
    sny, eny = cny - min_y // 2, cny - min_y // 2 + min_y
    snz, enz = cnz - min_z // 2, cnz - min_z // 2 + min_z

    kspace_new[snx:enx, sny:eny, snz:enz] = kspace[sx:ex, sy:ey, sz:ez]

    # Noise goes in BEFORE the window so it picks up the same spatial
    # correlation (apodization) as the signal.
    if complex_noise is not None:
        kspace_new = kspace_new + complex_noise

    if tukey_window is not None:
        kspace_new = kspace_new * tukey_window

    img_lowres = torch.fft.ifftn(torch.fft.ifftshift(kspace_new))

    # IFFT divides by N_new while the forward FFT did not divide by N_original,
    # so restore the original intensity scale.
    N_original = siz[0] * siz[1] * siz[2]
    N_new = nx * ny * nz
    img_lowres = img_lowres * (N_new / N_original)

    return torch.abs(img_lowres) if take_abs else img_lowres.real


def kspace_noise_std(noise_std, siz, newsize):
    """k-space noise sigma giving the requested image-space sigma.

    IFFT divides by N_new and the explicit rescale multiplies by
    N_new/N_original, so image-space noise variance ends up scaled by
    N_original^2 / N_new relative to what was injected.
    """
    N_original = siz[0] * siz[1] * siz[2]
    N_new = int(newsize[0]) * int(newsize[1]) * int(newsize[2])
    return noise_std * N_original / np.sqrt(N_new)


def _maybe_compact(result, compact_labels):
    """Replace the 56-channel one-hot segmentations with argmax indices.

    Applied at YIELD time, not when the result is built: the cheap-repeat path
    reuses the cached one-hot inside _correct_lesion_visibility, so compacting
    earlier would break it.

    Every consumer argmaxes these immediately, and at typical volume sizes
    the one-hot costs roughly 28x more memory per stream than compact
    indices. That matters most when cases sit in a prefetch queue, holding
    several volumes' worth of GPU memory alive at once.

    Deliberately different key names rather than the same keys holding
    different data, so anything that genuinely needs the soft one-hot raises
    KeyError instead of silently argmaxing an index tensor. uint8 is safe for
    56 labels.
    """
    if not compact_labels:
        return result
    out = dict(result)
    for src, dst in (('seg_clean_onehot', 'seg_clean_idx'),
                     ('seg_path_onehot', 'seg_path_idx')):
        if src in out:
            v = out.pop(src)
            # deform_labels_argmax already returned indices; only the one-hot
            # form still needs reducing.
            out[dst] = (v if v.dim() == 3 else torch.argmax(v, dim=-1)).to(torch.uint8)
    return out


def deform_labels_argmax(label_map, weights, n_labels, device):
    """Deform a label map and return argmax indices, without ever building the
    one-hot.

    Bit-exact with argmax(interp(onehot(labels))), because only the EIGHT
    corner voxels carry nonzero weight at any output location: accumulating
    those eight weights per label and taking the argmax is the same
    computation as blending all n_labels channels and taking the argmax, minus
    the channels that are zero everywhere.

    The gathers dominate this computation's cost -- large random lookups of
    224-byte rows are cache-hostile, far more expensive than the multiply
    that follows. This replaces eight 56-channel gathers with eight
    1-channel gathers plus one scatter_add, roughly an order of magnitude
    faster at typical volume sizes.
    """
    [ok, shape, fx, fy, fz, cx, cy, cz, wfx, wfy, wfz, wcx, wcy, wcz] = weights
    W = torch.stack([wfx * wfy * wfz, wcx * wfy * wfz, wfx * wcy * wfz, wcx * wcy * wfz,
                     wfx * wfy * wcz, wcx * wfy * wcz, wfx * wcy * wcz, wcx * wcy * wcz], 1)
    C = torch.stack([label_map[fx, fy, fz], label_map[cx, fy, fz],
                     label_map[fx, cy, fz], label_map[cx, cy, fz],
                     label_map[fx, fy, cz], label_map[cx, fy, cz],
                     label_map[fx, cy, cz], label_map[cx, cy, cz]], 1).long()
    acc = torch.zeros(C.shape[0], n_labels, dtype=W.dtype, device=device)
    acc.scatter_add_(1, C, W)
    out = torch.zeros(shape, dtype=torch.long, device=device)
    out[ok] = acc.argmax(1)
    return out


def apply_susceptibility(img, lesion_mask, device, chi_ppm=None, te_ms=None,
                         b0_t=3.0, dtype=torch.float32):
    """Signal dropout from a susceptibility source, via the dipole kernel.

    A susceptibility distribution chi perturbs the static field by a
    convolution with the dipole kernel, which is a pointwise multiply in
    k-space:

        dB(k) = (1/3 - kz^2/|k|^2) chi(k)

    so the whole field map costs one FFT, a multiply, and one inverse -- the
    same order as the k-space motion simulation, not a solver.

    The lesion mask is the source. Two effects follow, and only the first is
    modelled here: intravoxel dephasing, where signal decays with the field
    GRADIENT across a voxel rather than with the field itself (a uniform
    offset does not dephase). Geometric distortion along the readout is the
    other and is not applied.

    Magnitude is sampled rather than fixed because T1w susceptibility spans a
    wide range in practice. Spin-echo T1w refocuses static dephasing, so the
    effect is subtle; MPRAGE is gradient-echo based and blooms visibly around
    hemosiderin. Covering "nearly invisible" through "obvious dark blooming"
    is more faithful than picking one point, and matches how the other axes
    are drawn.

    This is the first place appearance depends on lesion GEOMETRY rather than
    being drawn independently of it -- P(X|Y) conditioned on the shape of Y,
    not just its label.
    """
    if lesion_mask is None or not lesion_mask.any():
        return img

    # ppm; hemosiderin and calcification sit well above brain tissue, and the
    # wide range keeps both near-invisible and strongly blooming cases in
    # support rather than fixing one severity.
    if chi_ppm is None:
        chi_ppm = 0.05 + 9.95 * torch.sigmoid(torch.randn(1, device=device)).item()
    if te_ms is None:
        te_ms = 2.0 + 18.0 * torch.rand(1, device=device).item()

    chi = lesion_mask.to(dtype) * (chi_ppm * 1e-6)
    D = chi.shape

    kz = torch.fft.fftfreq(D[0], device=device, dtype=dtype).view(-1, 1, 1)
    ky = torch.fft.fftfreq(D[1], device=device, dtype=dtype).view(1, -1, 1)
    kx = torch.fft.fftfreq(D[2], device=device, dtype=dtype).view(1, 1, -1)
    k2 = kz ** 2 + ky ** 2 + kx ** 2
    # The kernel is singular at k=0 (the dipole field of a uniform
    # susceptibility is undefined); zeroing the DC term drops the constant
    # offset, which does not dephase anyway.
    k2 = torch.where(k2 == 0, torch.ones_like(k2), k2)
    kernel = (1.0 / 3.0) - (kz ** 2) / k2
    kernel[0, 0, 0] = 0.0

    dB = torch.fft.ifftn(torch.fft.fftn(chi) * kernel).real   # in units of B0

    # Intravoxel dephasing goes with the field GRADIENT: a uniform offset
    # shifts phase without spreading it, so it costs no signal.
    gz, gy, gx = torch.gradient(dB)
    grad_mag = (gz ** 2 + gy ** 2 + gx ** 2).sqrt()

    gamma = 2.675e8                       # rad/s/T, proton
    decay = torch.exp(-gamma * b0_t * grad_mag * (te_ms * 1e-3))
    return img * decay


# GMM parameter table: the number of independently-drawn (mu, sigma) pairs
# available, i.e. how many tissue classes with separate intensity statistics
# one synthetic image can contain. Not a quantization of the output -- noise,
# bias field, blur and rescale are all continuous downstream.
GMM_TABLE_SIZE = 256

# Label partitioning within that table. 0-99 are anatomical labels from
# generation_labels.nii; PARTIAL_VOLUME_MAX is the top of the partial-volume
# band; whatever remains above it is free for temporary zone labels.
PARTIAL_VOLUME_MAX = 250
ZONE_G_LABEL_BASE = PARTIAL_VOLUME_MAX + 1          # 251
MAX_ZONE_LABELS = GMM_TABLE_SIZE - ZONE_G_LABEL_BASE  # 5, as an expression


def make_affine_matrix(rot, sh, s):
    Rx = np.array([[1, 0, 0], [0, np.cos(rot[0]), -np.sin(rot[0])], [0, np.sin(rot[0]), np.cos(rot[0])]])
    Ry = np.array([[np.cos(rot[1]), 0, np.sin(rot[1])], [0, 1, 0], [-np.sin(rot[1]), 0, np.cos(rot[1])]])
    Rz = np.array([[np.cos(rot[2]), -np.sin(rot[2]), 0], [np.sin(rot[2]), np.cos(rot[2]), 0], [0, 0, 1]])

    SHx = np.array([[1, 0, 0], [sh[1], 1, 0], [sh[2], 0, 1]])
    SHy = np.array([[1, sh[0], 0], [0, 1, 0], [0, sh[2], 1]])
    SHz = np.array([[1, 0, sh[0]], [0, 1, sh[1]], [0, 0, 1]])

    A = SHx @ SHy @ SHz @ Rx @ Ry @ Rz
    A[0, :] = A[0, :] * s[0]
    A[1, :] = A[1, :] * s[1]
    A[2, :] = A[2, :] * s[2]

    return A


def make_gaussian_kernel(sigma, device, dtype):
    sl = int(np.ceil(3 * sigma))
    ts = torch.linspace(-sl, sl, 2 * sl + 1, dtype=dtype, device=device)
    gauss = torch.exp((-(ts / sigma) ** 2 / 2))
    kernel = gauss / gauss.sum()

    return kernel


def gaussian_blur_3d(input, stds, device, dtype=torch.float):
    blurred = input[None, None, :, :, :]
    if stds[0] > 0:
        kx = make_gaussian_kernel(stds[0], device=device, dtype=dtype)
        blurred = conv3d(blurred, kx[None, None, :, None, None], stride=1, padding=(len(kx) // 2, 0, 0))
    if stds[1] > 0:
        ky = make_gaussian_kernel(stds[1], device=device, dtype=dtype)
        blurred = conv3d(blurred, ky[None, None, None, :, None], stride=1, padding=(0, len(ky) // 2, 0))
    if stds[2] > 0:
        kz = make_gaussian_kernel(stds[2], device=device, dtype=dtype)
        blurred = conv3d(blurred, kz[None, None, None, None, :], stride=1, padding=(0, 0, len(kz) // 2))
    return torch.squeeze(blurred)


def fast_3D_interp_torch_linear_given_weights(X, weights, mode, device, dtype=torch.float, default_value=0.0):
    """Trilinear resample using precomputed weights.

    Vectorized over channels. The eight corner gathers use the SAME indices
    for every channel -- only the data differs -- so a per-channel Python
    loop would repeat identical index arithmetic C times. That is invisible
    for a 1-channel volume but dominates for the 56-channel one-hot
    segmentation, where it would mean 56 iterations x 8 gathers = 448 large
    gather ops instead of 8.

    Indexing the full (D,H,W,C) tensor at [fx,fy,fz] gathers all channels in
    one op, so the weighting is broadcast over a trailing channel axis.
    """
    [ok, shape, fx, fy, fz, cx, cy, cz, wfx, wfy, wfz, wcx, wcy, wcz] = weights

    squeeze_out = (len(X.shape) == 3)
    if squeeze_out:
        X = X[..., None]

    # Fused trilinear blend. The naive form builds 14 intermediate (N, C)
    # tensors -- at N=250k, C=56 that is ~56 MB each, and profiling showed the
    # broadcast multiplies (not the gathers) were ~2.5s of a ~2.9s call.
    # Folding the three axis weights into 8 scalar corner weights first turns
    # that into 8 multiply-accumulates over (N, C). Reassociating the
    # multiplies changes float32 rounding by ~1e-7; it is not a logic change.
    w000 = (wfx * wfy * wfz)[:, None]
    w100 = (wcx * wfy * wfz)[:, None]
    w010 = (wfx * wcy * wfz)[:, None]
    w110 = (wcx * wcy * wfz)[:, None]
    w001 = (wfx * wfy * wcz)[:, None]
    w101 = (wcx * wfy * wcz)[:, None]
    w011 = (wfx * wcy * wcz)[:, None]
    w111 = (wcx * wcy * wcz)[:, None]

    c = X[fx, fy, fz] * w000
    c += X[cx, fy, fz] * w100
    c += X[fx, cy, fz] * w010
    c += X[cx, cy, fz] * w110
    c += X[fx, fy, cz] * w001
    c += X[cx, fy, cz] * w101
    c += X[fx, cy, cz] * w011
    c += X[cx, cy, cz] * w111

    Y = torch.full([*shape, X.shape[3]], default_value, dtype=dtype, device=device)
    Y[ok] = c.type(dtype)

    if squeeze_out or Y.shape[-1] == 1:
        Y = Y[..., 0]

    return Y


def fast_3D_interp_torch(X, II, JJ, KK, mode, device, default_value=0.0, dtype=torch.float, return_linear_weights=False):
    if mode == 'nearest':
        ok = (II >= 0) & (JJ >= 0) & (KK >= 0) & (II <= (X.shape[0] - 1)) & (JJ <= (X.shape[1] - 1)) & (KK <= (X.shape[2] - 1))
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        if len(X.shape) == 3:
            X = X[..., None]
        Y = torch.zeros([*II.shape, X.shape[3]], dtype=dtype, device=device)
        mask = (ok==False)
        for channel in range(X.shape[3]):
            aux = X[:, :, :, channel]
            Y[:, :, :, channel] = aux[IIr, JJr, KKr]
            Y[..., channel][mask] = default_value
        if Y.shape[3] == 1:
            Y = Y[:, :, :, 0]

    elif mode == 'linear':
        ok = (II >= 0) & (JJ >= 0) & (KK >= 0) & (II <= (X.shape[0] - 1)) & (JJ <= (X.shape[1] - 1)) & (KK <= (X.shape[2] - 1))
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]

        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = IIv - fx
        wfx = 1 - wcx

        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = JJv - fy
        wfy = 1 - wcy

        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = KKv - fz
        wfz = 1 - wcz

        if len(X.shape) == 3:
            X = X[..., None]

        Y = torch.zeros([*II.shape, X.shape[3]], dtype=dtype, device=device)
        for channel in range(X.shape[3]):
            Xc = X[:, :, :, channel]

            c000 = Xc[fx, fy, fz]
            c100 = Xc[cx, fy, fz]
            c010 = Xc[fx, cy, fz]
            c110 = Xc[cx, cy, fz]
            c001 = Xc[fx, fy, cz]
            c101 = Xc[cx, fy, cz]
            c011 = Xc[fx, cy, cz]
            c111 = Xc[cx, cy, cz]

            c00 = c000 * wfx + c100 * wcx
            c01 = c001 * wfx + c101 * wcx
            c10 = c010 * wfx + c110 * wcx
            c11 = c011 * wfx + c111 * wcx

            c0 = c00 * wfy + c10 * wcy
            c1 = c01 * wfy + c11 * wcy

            c = c0 * wfz + c1 * wcz

            Yc = torch.zeros(II.shape, dtype=dtype, device=device)
            Yc[ok] = c.type(dtype)
            Yc[~ok] = default_value
            Y[..., channel] = Yc

        if Y.shape[-1] == 1:
            Y = Y[..., 0]

    else:
        raise Exception('mode must be linear or nearest')

    if return_linear_weights:
        return Y, [ok, II.shape, fx, fy, fz, cx, cy, cz, wfx, wfy, wfz, wcx, wcy, wcz]
    else:
        return Y


def myzoom_torch(X, factor, device, aff=None):

    if len(X.shape)==3:
        X = X[..., None]

    dtype = X.dtype

    delta = (1.0 - factor) / (2.0 * factor)
    newsize = np.round(X.shape[:-1] * factor).astype(int)

    vx = torch.arange(delta[0], delta[0] + newsize[0] / factor[0], 1 / factor[0], dtype=dtype, device=device)[:newsize[0]]
    vy = torch.arange(delta[1], delta[1] + newsize[1] / factor[1], 1 / factor[1], dtype=dtype, device=device)[:newsize[1]]
    vz = torch.arange(delta[2], delta[2] + newsize[2] / factor[2], 1 / factor[2], dtype=dtype, device=device)[:newsize[2]]

    vx[vx < 0] = 0
    vy[vy < 0] = 0
    vz[vz < 0] = 0
    vx[vx > (X.shape[0]-1)] = (X.shape[0]-1)
    vy[vy > (X.shape[1] - 1)] = (X.shape[1] - 1)
    vz[vz > (X.shape[2] - 1)] = (X.shape[2] - 1)

    fx = torch.floor(vx).int()
    cx = fx + 1
    cx[cx > (X.shape[0]-1)] = (X.shape[0]-1)
    wcx = vx - fx
    wfx = 1 - wcx

    fy = torch.floor(vy).int()
    cy = fy + 1
    cy[cy > (X.shape[1]-1)] = (X.shape[1]-1)
    wcy = vy - fy
    wfy = 1 - wcy

    fz = torch.floor(vz).int()
    cz = fz + 1
    cz[cz > (X.shape[2]-1)] = (X.shape[2]-1)
    wcz = vz - fz
    wfz = 1 - wcz

    Y = torch.zeros([newsize[0], newsize[1], newsize[2], X.shape[3]], dtype=dtype, device=device)

    # Vectorised over output slices. Each of the three loops was a separable
    # 1D interpolation launching one kernel per output slice -- ~480 launches
    # per channel per call at 160^3, with 5.5 calls per draw. GPU profiling put
    # this at 32.8% of a fresh draw on the pathology arm.
    #
    # Advanced indexing on the axis being resampled does the same gather for
    # every slice at once, and the weights broadcast along the untouched axes.
    # Same arithmetic, same order of operations per output voxel, so the result
    # is bit-identical -- not an interpolation-scheme change. NOT F.interpolate,
    # which uses a different sampling-grid convention and would shift the output
    # by half a voxel.
    fx_l, cx_l = fx.long(), cx.long()
    fy_l, cy_l = fy.long(), cy.long()
    fz_l, cz_l = fz.long(), cz.long()

    # x: (nx, Y, Z, C)
    tmp1 = X[fx_l] * wfx.view(-1, 1, 1, 1) + X[cx_l] * wcx.view(-1, 1, 1, 1)
    # y: (nx, ny, Z, C)
    tmp2 = tmp1[:, fy_l] * wfy.view(1, -1, 1, 1) + tmp1[:, cy_l] * wcy.view(1, -1, 1, 1)
    # z: (nx, ny, nz, C)
    Y = tmp2[:, :, fz_l] * wfz.view(1, 1, -1, 1) + tmp2[:, :, cz_l] * wcz.view(1, 1, -1, 1)

    if Y.shape[3] == 1:
        Y = Y[:,:,:, 0]

    if aff is not None:
        aff_new = aff.copy()
        for c in range(3):
            aff_new[:-1, c] = aff_new[:-1, c] / factor
        aff_new[:-1, -1] = aff_new[:-1, -1] - aff[:-1, :-1] @ (0.5 - 0.5 / (factor * np.ones(3)))
        return Y, aff_new
    else:
        return Y
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
# import ext.interpol as interpol
from mcp_pathology_randomizer import ROSE_CRITERION_K
# Same constant used for both the label-level geometric gate
# (mcp_pathology_randomizer.py's augment_batch) and the image-level
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

        oasis_ids = _get_healthy_ids(os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_data_root') + '/OASIS3/participants.csv')
        adni_ids = _get_healthy_ids(os.environ.get('PR_EVAL_DATA_ROOT', '/path/to/eval_data_root') + '/ADNI3/participants.tsv')
        
        healthy_datasets_all = ['Buckner40', 'Chinese-HCP', 'HCP']
        healthy_datasets_partial = {'OASIS3': oasis_ids, 'ADNI3': adni_ids}
        
        active_datasets = []
        total_healthy_scans = 0
        
        for d in datasets:
            if d not in healthy_datasets_all and d not in healthy_datasets_partial:
                continue
                
            dataset_files = glob.glob(os.path.join(datadir, d + '.*' + 'T1w.nii'))
            
            if d in healthy_datasets_partial:
                valid_ids = healthy_datasets_partial[d]
                filtered = []
                for f in dataset_files:
                    file_clean = clean_id(os.path.basename(f))
                    # Check if the cleaned filename contains the cleaned ID
                    if any(hid in file_clean for hid in valid_ids if len(hid) > 3):
                        filtered.append(f)
                dataset_files = filtered
                print(f"  Filtered {d}: kept {len(dataset_files)} healthy scans.")
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

        # Vector to flip left and right
        nlat = int((n_labels - n_neutral_labels) / 2.0)
        vflip = np.concatenate([np.array(range(n_neutral_labels)),
                                np.array(range(n_neutral_labels + nlat, n_labels)),
                                np.array(range(n_neutral_labels, n_neutral_labels + nlat))])

        print('Generator is ready!')

    return types.SimpleNamespace(
        datasets=datasets, names=names, res_training_data=res_training_data,
        max_surf_dist_modeled=max_surf_dist_modeled, xx=xx, yy=yy, zz=zz, c=c,
        xc=xc, yc=yc, zc=zc, n_labels=n_labels, lut=lut, onehotmatrix=onehotmatrix,
        lut_normalization=lut_normalization, lut_kill_photo=lut_kill_photo,
        lut_kill_photo_inc_ventricles=lut_kill_photo_inc_ventricles, vflip=vflip,
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
    G = torch.squeeze(torch.tensor(Gimg.get_fdata()[x1:x2, y1:y2, z1:z2], dtype=torch.int, device=device))
    S = torch.squeeze(torch.tensor(nib.load(segmentation_labels).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=torch.int, device=device))
    T1 = torch.squeeze(torch.tensor(nib.load(t1).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))
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
    T1dm = None if (t1dm is None or _skip) else torch.squeeze(torch.tensor(nib.load(t1dm).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))
    T2 = None if (t2 is None or _skip) else torch.squeeze(torch.tensor(nib.load(t2).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))
    T2dm = None if (t2dm is None or _skip) else torch.squeeze(torch.tensor(nib.load(t2dm).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))
    FLAIR = None if (flair is None or _skip) else torch.squeeze(torch.tensor(nib.load(flair).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))
    FLAIRdm = None if (flairdm is None or _skip) else torch.squeeze(torch.tensor(nib.load(flairdm).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))
    CT = None if (ct is None or _skip) else torch.squeeze(torch.tensor(nib.load(ct).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))
    CTdm = None if (ctdm is None or _skip) else torch.squeeze(torch.tensor(nib.load(ctdm).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))
    D = 0.1 * torch.squeeze(torch.tensor(nib.load(brain_dist_map).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))
    LP = None if _skip else 0.05 * (torch.squeeze(torch.tensor(nib.load(lp_dist_map).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device)) - 128.0)
    RP = None if (rp_dist_map is None or _skip) else (0.05 * (torch.squeeze(torch.tensor(nib.load(rp_dist_map).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device)) - 128.0))
    LW = None if _skip else 0.05 * (torch.squeeze(torch.tensor(nib.load(lw_dist_map).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device)) - 128.0)
    RW = None if (rw_dist_map is None or _skip) else (0.05 * (torch.squeeze(torch.tensor(nib.load(rw_dist_map).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device)) - 128.0))
    MNIx = None if _skip else 0.0001 * torch.squeeze(torch.tensor(nib.load(mni_reg_x).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))
    MNIy = None if _skip else 0.0001 * torch.squeeze(torch.tensor(nib.load(mni_reg_y).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))
    MNIz = None if _skip else 0.0001 * torch.squeeze(torch.tensor(nib.load(mni_reg_z).get_fdata()[x1:x2, y1:y2, z1:z2], dtype=dtype, device=device))

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
    }


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
            mus = 25 + 200 * torch.rand(256, dtype=dtype, device=device)
            sigmas = 5 + 20 * torch.rand(256, dtype=dtype, device=device)
            # set the background to zero if photo
            if (mode == 'photo'):
                mus[0] = 0
                sigmas[0] = 0
            elif (np.random.rand() < 0.5):  # actually, let's do this frequently, but not with zero variance!
                mus[0] = 0

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

            SYN = mus[G] + sigmas[G] * torch.randn(G.shape, dtype=dtype, device=device)
            SYN[SYN < 0] = 0
            SYN /= mus[2]  # normalize by WM
            SYNblur = gaussian_blur_3d(SYN, 0.5 * np.ones(3), device, dtype=dtype)  # cosmetic

            # Make random linear combinations.
            # Default False reproduces SynthSeg, whose training data "does
            # not include any real scan". Set True for the original
            # behaviour, where half of all training images are a random
            # blend of the synthetic GMM image with the subject's REAL
            # T1/T2/FLAIR -- note this breaks the "no real scans" claim.
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


def _inject_pathology(S, G, device, reference_image, lesion_g_label=1, lesion_s_label=33,
                       mode_weights=(0.20, 0.15, 0.15, 0.25, 0.25), k_rose=ROSE_CRITERION_K,
                       n_lesion_zones=3):
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
    that check numerically arbitrary (see mcp_pathology_randomizer.py). The
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
    and fold_native: (D,H,W) bool, True where the mass-effect displacement
    locally folded (non-invertible Jacobian).
    """
    from mcp_pathology_randomizer import PathologyRandomizer

    shape = tuple(S.shape[-3:])
    aug = PathologyRandomizer((1, 1, *shape), device=device)

    S_native = S.reshape(1, 1, *shape).float()
    ref_native = reference_image.reshape(1, 1, *shape).float()
    out = aug.augment_batch(ref_native, seg_tensor=S_native, batch_size=1, mode_weights=mode_weights, k_rose=k_rose)

    relabel_mask = (out['seg_with_novel_label'] != out['warped_seg'])[0, 0]  # (D,H,W) bool

    S_path = out['warped_seg'][0, 0].clone()
    S_path[relabel_mask] = float(lesion_s_label)  # all zones collapse to 33 in S
    S_path = S_path.round().to(S.dtype)

    G_native = G.reshape(1, 1, *shape).float()
    G_warped = F.grid_sample(G_native, out['grid'], mode='nearest', padding_mode='border', align_corners=False)[0, 0]
    G_path = G_warped.clone()

    if n_lesion_zones <= 1 or not relabel_mask.any():
        # Degenerate case: single zone, original behaviour
        G_path[relabel_mask] = float(lesion_g_label)
    else:
        # Subdivide the lesion region into n_lesion_zones distinct GMM classes
        # using a spectral random field thresholded into equal-probability
        # quantile bins -- so zones are roughly equal in size regardless of
        # the field's exact distribution, and the zone boundaries are
        # spatially coherent (correlated at the same scale as the lesion's
        # own spatial structure).
        # Temporary G labels 251..(251+n_lesion_zones-1): above the partial-
        # volume band (100-250), below the 256-entry mus/sigmas table limit,
        # and not present in real generation_labels.nii files (which use only
        # low anatomical indices). These are ONLY written to G_path here and
        # used during GMM synthesis; they never appear in S_path or any
        # output the training loop or network sees.
        ZONE_G_LABEL_BASE = 251  # first temporary zone label
        zone_field = aug._universal_spectral_field(
            batch_size=1,
            max_freq=0.15,   # spatial scale ~6-7 voxels -- coarser than
                             # texture (which tracks lesion radius), since
                             # necrosis/rim/edema are large-scale zones, not
                             # fine-grained texture
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

    return S_path, G_path, out['metadata'], fold_native


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
                                min_noise_std=0.05,
                                max_noise_std=0.15,
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
                                k_rose_sampler=None):
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

    from mcp_pathology_randomizer import PathologyRandomizer  # needed here for the contrast-retry Rose check below

    def _cheap_perturb_pair(img_clean, img_path):
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

        return _apply(img_clean), _apply(img_path)

    def _correct_lesion_visibility(final_clean, final_path, seg_path_onehot, seg_clean_onehot, current_k_rose,
                                    stds, II, JJ, KK, factors, M_path):
        # Recomputed per call: cheap-repeat's own noise injection shifts the
        # noise floor after the label was decided, so tau can't be cached.
        lesion_mask_out = seg_path_onehot[..., lesion_compact_idx] == seg_path_onehot.max(dim=-1).values
        if not lesion_mask_out.any():
            return final_path
        aug_check = PathologyRandomizer((1, 1) + tuple(final_clean.shape), device=device)
        # tau = visibility floor. See ROSE_CRITERION_K in
        # mcp_pathology_randomizer.py for citation.
        tau = aug_check._min_visible_change(final_clean.reshape(1, 1, *final_clean.shape), k_rose=current_k_rose, fallback_noise_std_frac=0.05)
        visible_change_signed = final_path[lesion_mask_out] - final_clean[lesion_mask_out]
        mean_visible = visible_change_signed.abs().mean()
        if mean_visible <= tau:
            # Toward whichever side has more room -- guarantees >=0.5 room
            # at every voxel, regardless of the original draw's sign.
            sign = torch.where(final_clean[lesion_mask_out] < 0.5, 1.0, -1.0)

            # severity (one scalar/draw, overall conspicuousness) and
            # texture (a spatial field, internal heterogeneity) are two
            # independent multiplicative sources of variation. Both use
            # unscaled sigmoid(randn()) where (0,1) is the parameter's own
            # natural range.
            room = torch.where(sign > 0, 1.0 - final_clean[lesion_mask_out], final_clean[lesion_mask_out])
            # MAX_SEVERITY_MULTIPLE: design choice, not measured. Bounds the
            # PRE-chain severity target only -- Newton's method below can
            # scale it upward if the resolution chain dilutes it. room is a
            # safety cap against excessive headroom-chasing, not the target.
            MAX_SEVERITY_MULTIPLE = 2.0
            achievable_severity = min(MAX_SEVERITY_MULTIPLE, (room.median() / tau).item())
            SEVERITY_SPREAD = max(0.0, achievable_severity - 1.0)

            # Internal sub-structure size scales with lesion size, not a
            # fixed absolute.
            n_lesion_voxels = lesion_mask_out.sum().item()
            lesion_radius = (3 * n_lesion_voxels / (4 * math.pi)) ** (1 / 3)  # equivalent sphere radius, in voxels
            TEXTURE_SCALE_FRACTION = torch.sigmoid(torch.randn(1, device=device)).item()
            TEXTURE_SPATIAL_SCALE = max(1.0, TEXTURE_SCALE_FRACTION * lesion_radius)
            # Must stay <1: texture ranges over (1-SPREAD, 1+SPREAD), and
            # SPREAD>=1 lets texture go negative, flipping the correction's
            # sign at that voxel.
            TEXTURE_SPREAD = torch.sigmoid(torch.randn(1, device=device)).item()

            severity = 1.0 + SEVERITY_SPREAD * torch.sigmoid(torch.randn(1, device=device)).item()

            # Multi-scale field via _universal_spectral_field (also used for
            # geometric placement elsewhere). max_freq ~ 1/spatial_scale;
            # the function's own cap (~half Nyquist) handles tiny lesions.
            max_freq = 1.0 / TEXTURE_SPATIAL_SCALE
            smooth = aug_check._universal_spectral_field(batch_size=1, max_freq=max_freq, force_dc=False).squeeze(0).squeeze(0)
            texture = 1.0 + TEXTURE_SPREAD * (2 * torch.sigmoid(smooth) - 1)  # field is already zero-mean, unit-std

            target_diff = sign * tau * severity * texture[lesion_mask_out]

            # Run through the same blur/downsample/upsample chain the rest
            # of the image went through, rather than setting sharp values
            # directly. Injected after gamma/bias-field (upstream, nothing
            # to invert) so the remaining chain is exactly linear.
            chain_input = torch.zeros_like(final_clean)
            chain_input[lesion_mask_out] = target_diff * M_path
            blur_delta = gaussian_blur_3d(chain_input, stds, device, dtype=final_clean.dtype)
            small_delta = fast_3D_interp_torch(blur_delta, II, JJ, KK, 'linear', device, dtype=final_clean.dtype)
            resized_delta = myzoom_torch(small_delta, 1 / factors, device)
            # Chain(k*x) = k*Chain(x) since it's linear -- cache at k=1 and
            # rescale instead of re-running the chain per iteration.
            raw_delta_unit = resized_delta / M_path
            room = torch.where(raw_delta_unit >= 0, 1.0 - final_clean, final_clean) + 1e-6

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
            final_path = final_clean + room * tanh_val
            achieved = (final_path[lesion_mask_out] - final_clean[lesion_mask_out]).abs().mean()
            if achieved <= tau:
                for _ in range(MAX_NEWTON_ITERS):
                    sech2 = 1.0 - tanh_val ** 2
                    f_prime = (raw_delta_unit[lesion_mask_out].abs() * sech2[lesion_mask_out]).mean() + 1e-8
                    k = k + ((target_visible - achieved) / f_prime).item()
                    tanh_val = torch.tanh(k * raw_delta_unit / room)
                    final_path = final_clean + room * tanh_val
                    achieved = (final_path[lesion_mask_out] - final_clean[lesion_mask_out]).abs().mean()
                    if achieved >= target_visible:
                        break
                else:
                    # Exhausted iterations: room is smaller than tau at
                    # enough voxels that the mean can't be reached. Fall
                    # through to per-voxel suppression below.
                    pass

        # Per-voxel label suppression: applied unconditionally after the
        # final result is determined, whether Newton succeeded, exhausted,
        # or the initial k=1 attempt already passed the mean check. The
        # mean passing tau doesn't mean every voxel does -- a labeled voxel
        # where |final_path - final_clean| <= tau has no individually
        # visible signal, so claiming it as Lesion is false supervision.
        # ONLY the label reverts: the image keeps whatever the GMM
        # synthesis produced, since that signal went through the full
        # gamma/bias-field/resolution chain and is genuine synthesized
        # tissue. Reverting the image too would punch flat patches of clean
        # values into pathological surroundings -- a worse artifact than
        # the one being fixed. Modifies seg_path_onehot in-place through
        # the tensor reference -- propagates to the caller automatically.
        per_voxel_visible = (final_path[lesion_mask_out] - final_clean[lesion_mask_out]).abs()
        invisible = per_voxel_visible <= tau
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
        return final_path
        return final_path

    _repeat_counter = 0
    _cached = None

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
                cheap_result['image_clean'], cheap_result['image_path'] = _cheap_perturb_pair(
                    _cached['image_clean'], _cached['image_path'])
                cheap_result['image_path'] = _correct_lesion_visibility(
                    cheap_result['image_clean'], cheap_result['image_path'], cheap_result['seg_path_onehot'], cheap_result['seg_clean_onehot'], _cached['k_rose'],
                    _cached['stds'], _cached['II'], _cached['JJ'], _cached['KK'], _cached['factors'], _cached['M_path'])
                cheap_result['is_cheap_repeat'] = True
                yield cheap_result
                continue

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
            else:
              max_inject_retries = 8
              for _retry in range(max_inject_retries):
                S_path, G_path, pathology_metadata, fold_native = _inject_pathology(
                    S_clean, G_clean, device, T1, lesion_g_label=lesion_g_label, lesion_s_label=lesion_s_label,
                    mode_weights=mode_weights, k_rose=current_k_rose)
                if not torch.equal(S_path, S_clean):
                    break
            # else: exhausted retries (astronomically rare) -- proceed with
            # the last attempt rather than hang.

            # --- everything from here shares randomness between both passes ---
            # Visibility is checked on the FINAL synthesized image, not just
            # the label decision -- contrast is randomized independently of
            # geometry, so a Rose-passing LABEL can still land on a
            # randomly-drawn contrast where Lesion's mu sits close to its
            # neighbor's. Only Lesion voxels are checked; healthy anatomy is
            # left to blend untouched. See _correct_lesion_visibility below.
            lesion_compact_idx = label_list_segmentation.index(lesion_s_label)
            mus = 25 + 200 * torch.rand(256, dtype=dtype, device=device)
            sigmas = 5 + 20 * torch.rand(256, dtype=dtype, device=device)
            if mode == 'photo':
                mus[0] = 0
                sigmas[0] = 0
            elif np.random.rand() < 0.5:
                mus[0] = 0

            v_pv = 0.02 * torch.arange(50).to(device)
            mus[100:150] = mus[1] * (1 - v_pv) + mus[2] * v_pv
            mus[150:200] = mus[2] * (1 - v_pv) + mus[3] * v_pv
            mus[200:250] = mus[3] * (1 - v_pv) + mus[4] * v_pv
            mus[250] = mus[4]
            sigmas[100:150] = torch.sqrt(sigmas[1]**2 * (1 - v_pv) + sigmas[2]**2 * v_pv)
            sigmas[150:200] = torch.sqrt(sigmas[2]**2 * (1 - v_pv) + sigmas[3]**2 * v_pv)
            sigmas[200:250] = torch.sqrt(sigmas[3]**2 * (1 - v_pv) + sigmas[4]**2 * v_pv)
            sigmas[250] = sigmas[4]

            # SAME per-voxel synthesis noise for both -- voxels where G didn't
            # change get numerically identical synthesized intensity
            syn_noise = torch.randn(G_clean.shape, dtype=dtype, device=device)

            def _synth(G):
                SYN = mus[G] + sigmas[G] * syn_noise
                SYN[SYN < 0] = 0
                SYN = SYN / mus[2]
                return gaussian_blur_3d(SYN, 0.5 * np.ones(3), device, dtype=dtype)

            SYNblur_clean = _synth(G_clean)
            SYNblur_path = SYNblur_clean if not generate_pathology else _synth(G_path)

            # See mix_real_images note in super_generator -- applied
            # identically to both the clean and pathological stream.
            if mix_real_images and np.random.rand() < 0.5:
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
                Sonehot_clean = ctx.onehotmatrix[ctx.lut[S_clean]]
                SdefOneHot_clean = fast_3D_interp_torch_linear_given_weights(Sonehot_clean, lin_weights, 'linear', device, dtype=dtype)
                SdefOneHot_clean[..., 0] = fast_3D_interp_torch_linear_given_weights(Sonehot_clean[..., 0], lin_weights, 'linear', device, default_value=1.0, dtype=dtype)

                Sonehot_path = ctx.onehotmatrix[ctx.lut[S_path]]
                SdefOneHot_path = fast_3D_interp_torch_linear_given_weights(Sonehot_path, lin_weights, 'linear', device, dtype=dtype)
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

            # Model Resolution (SAME stds for both)
            stds = (0.85 + 0.3 * np.random.rand()) * np.log(5) / np.pi * thickness
            stds[thickness <= 1.0] = 0.0
            HRblur_clean = gaussian_blur_3d(HRbf_clean, stds, device, dtype=dtype)
            HRblur_path = gaussian_blur_3d(HRbf_path, stds, device, dtype=dtype)
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

            small_clean = fast_3D_interp_torch(HRblur_clean, II, JJ, KK, 'linear', device, dtype=dtype)
            small_path = fast_3D_interp_torch(HRblur_path, II, JJ, KK, 'linear', device, dtype=dtype)

            # SAME acquisition-noise draw for both -- this stage represents
            # scanner noise, which is not tissue-dependent, so sharing it is
            # more physically apt than drawing independently
            noise_std = torch.tensor(min_noise_std + (max_noise_std - min_noise_std) * np.random.rand(1), dtype=dtype, device=device)
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

            # Called before the flip so the delta-chain blur/resample
            # operates in the same coordinate system stds/II/JJ/KK/factors
            # were computed in.
            if generate_pathology:
              final_path = _correct_lesion_visibility(final_clean, final_path, SdefOneHot_path, SdefOneHot_clean, current_k_rose,
                                                      stds, II, JJ, KK, factors, M_path)

            # SAME flip decision for both
            if flipping and (np.random.rand() < 0.5):
                final_clean = torch.flip(final_clean, [0])
                SdefOneHot_clean = torch.flip(SdefOneHot_clean, [0])[:, :, :, ctx.vflip]
                final_path = torch.flip(final_path, [0])
                SdefOneHot_path = torch.flip(SdefOneHot_path, [0])[:, :, :, ctx.vflip]
                fold_def = torch.flip(fold_def, [0])

            result = {
                'image_clean': final_clean,
                'seg_clean_onehot': SdefOneHot_clean,
                'image_path': final_path,
                'seg_path_onehot': SdefOneHot_path,
                'pathology_metadata': pathology_metadata,
                'fold_mask': fold_def,  # (D,H,W) bool, output space -- True where the mass-effect
                                          # warp locally folded (see _mass_effect_jacobian_determinant)
                'mode': mode,
                'is_cheap_repeat': False,
                'k_rose': current_k_rose,  # Save the applied threshold for the visualizer/caching
                # Cached for the cheap-repeat branch's own
                # _correct_lesion_visibility call -- this case's resolution-
                # degradation character, shared across its cheap-repeats.
                'stds': stds, 'II': II, 'JJ': JJ, 'KK': KK, 'factors': factors, 'M_path': M_path,
            }
            _cached = result
            _repeat_counter = cheap_repeat_k - 1  # this yield is the 1st of cheap_repeat_k total
            yield result


#############




#######################
# Auxiliary functions #
#######################

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
    for every channel -- only the data differs -- so the original per-channel
    Python loop repeated identical index arithmetic C times. That is invisible
    for a 1-channel volume but dominates for the 56-channel one-hot
    segmentation: 56 iterations x 8 gathers = 448 large gather ops, measured
    at ~88% of a fresh draw's total cost.

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

    for channel in range(X.shape[3]):
        Xc = X[:,:,:,channel]

        tmp1 = torch.zeros([newsize[0], Xc.shape[1], Xc.shape[2]], dtype=dtype, device=device)
        for i in range(newsize[0]):
            tmp1[i, :, :] = wfx[i] * Xc[fx[i], :, :] +  wcx[i] * Xc[cx[i], :, :]
        tmp2 = torch.zeros([newsize[0], newsize[1], Xc.shape[2]], dtype=dtype, device=device)
        for j in range(newsize[1]):
            tmp2[:, j, :] = wfy[j] * tmp1[:, fy[j], :] +  wcy[j] * tmp1[:, cy[j], :]
        for k in range(newsize[2]):
            Y[:, :, k, channel] = wfz[k] * tmp2[:, :, fz[k]] +  wcz[k] * tmp2[:, :, cz[k]]

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
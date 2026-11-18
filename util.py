import os
import numpy as np
import glob
import nibabel as nib
import numpy as np
import torch


def generate_base_phantom(N=128):
    x = np.linspace(-1, 1, N)
    y = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, y)
    
    masks = {tissue: np.zeros((N, N), dtype=bool) for tissue in ['skull', 'gm', 'wm', 'csf']}
    
    ellipses = [
        (0, 0, 0.69, 0.92, 0, 'skull'),
        (0, -0.0184, 0.6624, 0.874, 0, 'gm'),
        (0.22, 0, 0.11, 0.31, -18, 'wm'),
        (-0.22, 0, 0.16, 0.41, 18, 'wm'),
        (0, 0.35, 0.21, 0.25, 0, 'csf'),
        (0, 0.1, 0.046, 0.046, 0, 'csf'),
        (0, -0.1, 0.046, 0.046, 0, 'csf')
    ]
    
    for cx, cy, rx, ry, angle, tissue in ellipses:
        rad = np.deg2rad(angle)
        X_rot = (X - cx) * np.cos(rad) + (Y - cy) * np.sin(rad)
        Y_rot = -(X - cx) * np.sin(rad) + (Y - cy) * np.cos(rad)
        mask = (X_rot**2 / rx**2 + Y_rot**2 / ry**2) <= 1
        masks[tissue][mask] = True
        
    masks['gm'][masks['wm'] | masks['csf']] = False
    return X, Y, masks['skull'], masks['gm'], masks['wm'], masks['csf']
def update_frame(frame, axes, ims, mgre_kspace, mgre_images, fse_kspace, fse_images, global_max):
    im_mgre_k, im_mgre_recon, im_fse_k, im_fse_recon = ims
    
    # Refresh axes
    for row in range(2):
        axes[row, 0].cla()
        axes[row, 0].set_facecolor('black')
    
    t = np.linspace(0, 6, 600)
    
    # --- Row 1: mGRE Profile ---
    rf_mgre = np.zeros_like(t)
    rf_mgre[(t > 0.2) & (t < 0.5)] = 0.6 * np.sin(2 * np.pi * 5 * t[(t > 0.2) & (t < 0.5)])
    axes[0, 0].plot(t, rf_mgre, color='tab:orange', label='RF Excitation')
    axes[0, 0].plot(t, np.exp(-0.6 * t), color='tab:gray', linestyle='--', label='T2* Decay')
    for i in range(1, 6):
        echo_center = i * 1.0 + 0.2
        color = 'tab:pink' if i - 1 == frame else 'tab:cyan'
        axes[0, 0].fill_between(t, 0, np.exp(-0.6 * t) * np.exp(-20*(t-echo_center)**2), color=color, alpha=0.5)
    axes[0, 0].set_title("mGRE Profile", fontsize=10)
    axes[0, 0].set_ylim(-0.5, 1.2)
    
    # --- Row 2: FSE Profile (The RF Pulses!) ---
    rf_fse = np.zeros_like(t)
    rf_fse[(t > 0.1) & (t < 0.4)] = 0.5 * np.sin(2 * np.pi * 5 * t[(t > 0.1) & (t < 0.4)])
    for i in range(1, 6):
        pc = i * 1.0 - 0.3 # Pulse centers for refocusing
        rf_fse[(t > pc-0.15) & (t < pc+0.15)] = 0.8 * np.sin(2 * np.pi * 8 * t[(t > pc-0.15) & (t < pc+0.15)])
    axes[1, 0].plot(t, rf_fse, color='tab:orange', label='RF Pulses')
    axes[1, 0].plot(t, np.exp(-0.15 * t), color='tab:gray', linestyle='--', label='T2 Decay')
    for i in range(1, 6):
        echo_center = i * 1.0 + 0.2
        color = 'tab:pink' if i - 1 == frame else 'tab:cyan'
        axes[1, 0].fill_between(t, 0, np.exp(-0.15 * t) * np.exp(-20*(t-echo_center)**2), color=color, alpha=0.5)
    axes[1, 0].set_title("FSE Profile", fontsize=10)
    axes[1, 0].set_ylim(-0.5, 1.2)
    
    # Update Images
    im_mgre_k.set_array(mgre_kspace[frame])
    im_mgre_k.set_clim(np.min(mgre_kspace[frame]), np.max(mgre_kspace[frame]))
    im_mgre_recon.set_array(mgre_images[frame] / global_max)
    
    im_fse_k.set_array(fse_kspace[frame])
    im_fse_k.set_clim(np.min(fse_kspace[frame]), np.max(fse_kspace[frame]))
    im_fse_recon.set_array(fse_images[frame] / global_max)
    
    return [im_mgre_k, im_mgre_recon, im_fse_k, im_fse_recon]

def make_mode_sampler(restricted=False, min_res=1.0, max_res_iso=4.0, max_res_aniso=8.0):
    """Build a mode_sampler.

    restricted=False (default) follows SynthSeg's randomise_res: each case is
    either fully isotropic low-res, U(min_res, max_res_iso) applied to all
    three axes, or anisotropic, where ONE random axis is sampled from
    U(min_res, max_res_aniso) and the other two stay at min_res. Slice
    thickness equals the sampled resolution, matching "blurred at the sampled
    LR, downsampled at LR".

    restricted=True is always anisotropic: one axis at U(2.5, 8.5) with
    thickness capped at U(4, 6), the other two axes fixed at 1mm. Never
    isotropic, and never mildly anisotropic (the low-res axis is always at
    least 2.5mm).

    Caveat: the isotropic/anisotropic split is inferred from SynthSeg's
    training.py docstring, which documents both max_res_iso and
    max_res_aniso but not how they are combined. The 50/50 choice here is a
    reading of that, not something verified against lab2im's source.
    """
    def sampler():
        mode = 'invivo'
        if restricted:
            resolution = np.array([1.0, 1.0, 1.0])
            thickness = np.array([1.0, 1.0, 1.0])
            idx = np.random.randint(3)
            resolution[idx] = 2.5 + 6 * np.random.rand()
            thickness[idx] = np.min([resolution[idx], 4.0 + 2.0 * np.random.rand()])
            return mode, resolution, thickness

        if np.random.rand() < 0.5:
            r = min_res + (max_res_iso - min_res) * np.random.rand()
            resolution = np.array([r, r, r])
        else:
            resolution = np.array([min_res, min_res, min_res])
            idx = np.random.randint(3)
            resolution[idx] = min_res + (max_res_aniso - min_res) * np.random.rand()
        return mode, resolution, resolution.copy()

    return sampler


# Default matches SynthSeg. Use make_mode_sampler(restricted=True) for
# always-anisotropic sampling instead.
mode_sampler = make_mode_sampler(restricted=False)

def get_slices(tensor_img):
    img_np = tensor_img.detach().cpu().numpy()
    mid_x, mid_y, mid_z = np.array(img_np.shape) // 2
    return [
        np.rot90(np.take(img_np, mid_x, axis=0), k=-1), # Sagittal
        np.rot90(np.take(img_np, mid_y, axis=1), k=-1), # Coronal
        np.rot90(np.take(img_np, mid_z, axis=2), k=-1)  # Axial
    ]

def sample_param_maps(datadir=os.environ.get('PR_DATA_DIR', '/path/to/data_dir') + '/', 
                           n_samples=5, 
                           device='cuda', 
                           save_to_disk=False, 
                           outputdir='/tmp/'):
    
    device = device if torch.cuda.is_available() else 'cpu'
    SEQUENCES = ['spgr', 'se', 'ir_se', 'bssfp', 'mprage']
    
    R1_list = sorted(glob.glob(datadir + '/*.R1.nii'))
    R2s_list = sorted(glob.glob(datadir + '/*.R2star.nii'))
    PD_list = sorted(glob.glob(datadir + '/*.PD.nii'))
    
    generated_results = []
    rng = np.random.default_rng()

    for n in range(n_samples):
        print(f'Sampling image {n+1} of {n_samples}')
        s = rng.integers(len(SEQUENCES))
        seq_name = SEQUENCES[s]
        idx = rng.integers(len(R1_list))

        img_R1 = nib.load(R1_list[idx])
        affine = img_R1.affine
        R1 = torch.tensor(img_R1.get_fdata(), device=device, dtype=torch.float) / 30000.0
        R2s = torch.tensor(nib.load(R2s_list[idx]).get_fdata(), device=device, dtype=torch.float) / 5000.0
        PD = torch.tensor(nib.load(PD_list[idx]).get_fdata(), device=device, dtype=torch.float)

        B1 = float(np.random.uniform(0.3, 7.0))
        if seq_name == 'spgr':
            theta = dict(TR=np.random.uniform(5e-3, 3.0), TE=np.random.uniform(1e-3, 50e-3), FA=np.random.uniform(5, 90), B1=B1)
        elif seq_name == 'se':
            theta = dict(TR=np.random.uniform(1.5, 7.0), TE=np.random.uniform(5e-3, 150e-3), B1=B1)
        elif seq_name == 'ir_se':
            theta = dict(TR=np.random.uniform(2.0, 7.0), TI=np.random.uniform(0.2, 3.0), TE=np.random.uniform(5e-3, 150e-3), B1=B1)
        elif seq_name == 'bssfp':
            theta = dict(TR=np.random.uniform(2e-3, 6e-3), FA=np.random.uniform(15, 75), B1=B1)
        elif seq_name == 'mprage':
            theta = dict(TR_inv=np.random.uniform(1.5, 6.0), TR=np.random.uniform(5e-3, 15e-3), 
                         TI=np.random.uniform(0.5, 1.5), TE=np.random.uniform(1e-3, 10e-3), FA=np.random.uniform(5, 15), B1=B1)

        eps = 1e-6
        T1, T2s = 1.0 / R1.clamp(min=eps), 1.0 / R2s.clamp(min=eps)

        if seq_name == 'spgr':
            FA_rad = torch.tensor(theta['FA'] * np.pi / 180.0, device=device)
            E1 = torch.exp(-theta['TR'] / T1)
            X = PD * B1 * torch.sin(FA_rad) * (1.0 - E1) / (1.0 - E1 * torch.cos(FA_rad) + eps) * torch.exp(-theta['TE'] / T2s)
        elif seq_name == 'se':
            X = PD * B1 * (1.0 - torch.exp(-theta['TR'] / T1)) * torch.exp(-theta['TE'] / T2s)
        elif seq_name == 'ir_se':
            X = PD * B1 * torch.abs(1.0 - 2.0 * torch.exp(-theta['TI'] / T1) + torch.exp(-theta['TR'] / T1)) * torch.exp(-theta['TE'] / T2s)
        elif seq_name == 'bssfp':
            FA_rad = torch.tensor(theta['FA'] * np.pi / 180.0, device=device)
            E1, E2 = torch.exp(-theta['TR'] / T1), torch.exp(-theta['TR'] / T2s)
            X = PD * B1 * torch.sin(FA_rad) * (1.0 - E1) / (1.0 - E1 * torch.cos(FA_rad) - E2 * (1.0 - torch.cos(FA_rad)) + eps)
        elif seq_name == 'mprage':
            FA_rad = torch.tensor(theta['FA'] * np.pi / 180.0, device=device)
            E_TR = torch.exp(-theta['TR'] / T1)
            inv_c = torch.abs(1.0 - 2.0 * torch.exp(-theta['TI'] / T1) + torch.exp(-theta['TR_inv'] / T1))
            readout = torch.sin(FA_rad) * (1.0 - E_TR) / (1.0 - E_TR * torch.cos(FA_rad) + eps)
            X = PD * B1 * inv_c * readout * torch.exp(-theta['TE'] / T2s)

        X_np = X.detach().cpu().numpy()
        generated_results.append({'data': X_np, 'seq': seq_name})

        if save_to_disk:
            nib.save(nib.Nifti1Image(X_np, affine), f"{outputdir}/sample_{n+1}.nii.gz")

    print('Processing complete.')
    return generated_results
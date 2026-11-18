import torch
import torch.fft as fft

def shear_x_kspace(kspace, s, device):
    """Applies a shear in X-direction: x -> x + s*y"""
    # Use shape[0] to make it robust to transpositions
    h = kspace.shape[0] 
    w = kspace.shape[1]
    
    # Grid for the Y-coordinate
    grid_y = torch.linspace(-0.5, 0.5, w, device=device).view(1, -1, 1)
    # The shift is a function of Y: s * Y
    shift_map = s * grid_y * h 
    
    # Phase shift: exp(-2j * pi * kx * shift_map)
    kx = torch.linspace(-0.5, 0.5, h, device=device).view(-1, 1, 1)
    phase = torch.exp(-2j * torch.pi * kx * shift_map)
    return kspace * phase

def rotate_kspace_fast(kspace, angle_deg, device):
    """Applies a 2D rotation in the XY plane using 3 shears."""
    if angle_deg == 0:
        return kspace
        
    theta = torch.deg2rad(torch.tensor(angle_deg, device=device))
    a = torch.tan(theta / 2)
    b = -torch.sin(theta)
    
    # 1. Shear X
    kspace = shear_x_kspace(kspace, a, device)
    # 2. Shear Y (transpose, shear X, transpose back)
    kspace = torch.transpose(kspace, 0, 1)
    kspace = shear_x_kspace(kspace, b, device)
    kspace = torch.transpose(kspace, 0, 1)
    # 3. Shear X
    kspace = shear_x_kspace(kspace, a, device)
    
    return kspace

def simulate_kspace_motion(image, device, num_segments=8, pe_axis=0, rot_angle=0.0):
    """
    Simulates motion artifacts (Translation + Rotation) in k-space.
    rot_angle: Degrees of rotation in the XY plane.
    """
    h, w, d = image.shape
    # FFT to k-space
    kspace = fft.fftn(image, dim=(0, 1, 2))
    kspace = fft.fftshift(kspace, dim=(0, 1, 2))
    
    # 1. Apply global rotation first (optional)
    if rot_angle != 0:
        kspace = rotate_kspace_fast(kspace, rot_angle, device)
    
    corrupted_kspace = torch.zeros_like(kspace)
    
    # Get dimension size of the chosen PE axis
    dim_size = kspace.shape[pe_axis]
    segment_size = dim_size // num_segments
    
    # Create 3D frequency grids
    grid_x = torch.linspace(-0.5, 0.5, h, device=device).view(-1, 1, 1)
    grid_y = torch.linspace(-0.5, 0.5, w, device=device).view(1, -1, 1)
    grid_z = torch.linspace(-0.5, 0.5, d, device=device).view(1, 1, -1)
    
    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size if i != num_segments - 1 else dim_size
        
        # Random 3D translation
        shift = torch.randn(3, device=device) * 2.0
        
        # Apply 3D phase shift
        phase_map = -2j * torch.pi * (grid_x * shift[0] + grid_y * shift[1] + grid_z * shift[2])
        phase_shift = torch.exp(phase_map)
        
        # Dynamic slicing
        slices = [slice(None)] * 3
        slices[pe_axis] = slice(start, end)
        
        # Apply shift
        corrupted_kspace[tuple(slices)] = kspace[tuple(slices)] * phase_shift[tuple(slices)]
        
    # Reconstruct
    corrupted_kspace = fft.ifftshift(corrupted_kspace, dim=(0, 1, 2))
    final_image = fft.ifftn(corrupted_kspace, dim=(0, 1, 2)).abs()
    return final_image


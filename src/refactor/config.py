
from dataclasses import dataclass 
from typing import Tuple

# Intra-frame codec (DMCI) 
class DMCIConfig:
    patch_size: int = 8                    # square patch size used in (un)shuffle
    src:        int = 3 * 8 * 8            # channels after pixel-unshuffle (3 × P²)

    enc_dec:   int = 368                   # width of encoder / decoder feature maps
    N:         int = 256                   # latent-space channel count (y)
    z_channel: int = 128                   # hyper-prior latent channels (z)

# Inter-frame codec (DMC)
@dataclass
class DMCConfig:
    patch_size: int = 8                    # square patch size used in (un)shuffle
    src:        int = 3 * 8 * 8            # channels after pixel-unshuffle (3 × P²)

    ch_d:     int = 256                    # main backbone feature width
    ch_y:     int = 128                    # latent-space channel count (y)
    ch_z:     int = 128                    # hyper-prior latent channels (z)
    ch_recon: int = 320                    # width of reconstruction network

    qp_shift:   Tuple[int, int, int] = (0, 8, 4)  # qp shift
    extra_qp: int = 8                            # largest shift (max(qp_shift))

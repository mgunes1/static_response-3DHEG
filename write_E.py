import matplotlib.pyplot as plt
import scienceplots  # noqa

plt.style.use(["science"])

# --- Organized modules ---
from utils.io_utils import (
    get_E_all,
)

dir_prefix = "v5.0"
#main_dir,rs,Ne are to be changed accordingly.
main_dir = f"/mnt/ceph/users/mgunes/HEG_dmc/static_response/{dir_prefix}/runs/rs5.0-n54/"

rs = 5
fig, ax = plt.subplots(dpi=200)
ecut_pre = 125
Ne = 54

get_E_all(main_dir, rs, Ne, prefix=dir_prefix + "_")

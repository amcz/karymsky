import glob
import os

import numpy as np



def get_volcat_1h(tdir, d1, d2):
    # get all VOLCAT Files between two dates
    if os.path.exists(tdir):
        #drange = None
        flist = glob.glob(tdir + "*nc")
        das = volcat.get_volcat_list(
            tdir, flist=None, verbose=True, daterange=[d1, d2], include_last=False
        )
        if len(das) == 0:
            print("NO VOLCAT FOUND")
            return [], None, None
        dset = volcat.combine_regridded(das)
        dset_averaged = dset.ash_mass_loading.mean(dim="time")
        dset_ht = dset.ash_cloud_height.max(dim="time")
        return das, dset_averaged, dset_ht


def volcat_mass(tdir, d1, d2):
    print(tdir)
    das, dset, dset_ht = get_volcat_1h(tdir, d1, d2)
    total_mass = []

    for d in das:
        total_mass.append(float(d.ash_mass_loading_total_mass.values))

    print("Total mass volcat", np.mean(total_mass))
    # print('Max mass loading volcat', np.max(dset))
    # print(np.nanmin(dset))
    # another way which multiplies area by mass loading
    # field in data array must be in g/m2.
    # returns mass in Tg
    # print(volcat.check_total_mass(dset))
    return dset, dset_ht

import os
from glob import glob

def clean(path):
    path = path.replace('$','')
    path = path.replace('\r','')
    return path

def purge(paths):
    for p in paths:
        p_ = clean(p)
        if p == p_:
            continue
        print(f"change needed! {p}->{p_}")
        os.rename(p, p_)

# Fix Paths
PATH_ROOT = "generated_panorama_run_2"
regex_generated_panoramas = PATH_ROOT+'/hdr/*'
paths_generated_panoramas = glob(regex_generated_panoramas)
purge(paths_generated_panoramas)
regex_generated_panoramas = PATH_ROOT+'/holistic/*'
paths_generated_panoramas = glob(regex_generated_panoramas)
purge(paths_generated_panoramas)

regex_generated_panoramas = PATH_ROOT+'PATH_ROOT/ldr/*'
paths_generated_panoramas = glob(regex_generated_panoramas)
purge(paths_generated_panoramas)
# SEEG dataset (not included in this repo)

This folder holds the per-subject iEEG NetCDF files used by
`../EITN_ebrains_school_oct_2025.ipynb`. The data (~430 MB, 59 subjects) is
not committed to git — download it manually from EBRAINS:

- Dataset: Lachaux, J.-P., Rheims, S., Chatard, B., Dupin, M., & Bertrand, O.
  (2023). Human Intracranial Database (release-5). EBRAINS.
  https://doi.org/10.25493/FCPJ-NZ
- Knowledge Graph entry:
  https://search.kg.ebrains.eu/instances/bb13d2d1-4609-4790-a20b-678836ad486f

After downloading, place the `.nc` files directly in this directory, keeping
the original naming convention:

```
HID-Sub-001_f50f150-sm0_nbefore-128_nafter-256.nc
HID-Sub-002_f50f150-sm0_nbefore-128_nafter-256.nc
...
```

The notebook loads every file in this directory with `os.listdir(root)`, so
no further configuration is needed once the files are in place.

# optlnls

Useful python functions for x-ray optics simulations and metrology developed in LNLS, CNPEM

### About hybrid:

Hybrid documentation and license can be found [HERE](optlnls/hybrid_funcs).

### For Developers: Publishing a New Package Version
Make sure you have `twine` installed:
```
pip install twine
```

Update the version number in `setup.py`.   
Then, on the root directory of the project, build and upload a new version to PyPI:
```
rm -fR dist/*
python3 setup.py sdist
python3 -m twine upload dist/*
```

## Authors:

Sergio A. Lordano Luiz (sergiolordano2@gmail.com)

Artur C. Pinto (artur.pinto@lnls.br)

Humberto Rigamonti Jr. (humberto.junior@lnls.br)

João Pedro I. Astolfo (joao.astolfo@lnls.br)

Bernd C. Meyer


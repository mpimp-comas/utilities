# Various Project Utilities

This repo is a collection of various common utilities for working with Python, chemical structures and databases.

## Content

### Notebooks
* [sqlite_sss.ipynb](notebooks/sqlite_sss.ipynb): Create a local SQLite database from Pandas dataframes and extend with a function for substructure searchin (apahl; Notebook)

### Python Scripts
* [stand_struct.py](python_scripts/stand_struct.py): A Python script for standardizing structure files. The input can be either SD files OR CSV or TSV files which contain the structures as `Smiles`. 
The output is always a TSV file, various options are available, please have a look at the help in the file. (apahl; Python script)

#### Usage

```
$ ./stand_struct.py --help
usage: stand_struct [-h] [--nocanon] [--min_heavy_atoms MIN_HEAVY_ATOMS] [--max_heavy_atoms MAX_HEAVY_ATOMS] [-d] [-c COLUMNS]
                    in_file {full,fullrac,medchem,medchemrac}

Standardize structures. Input files can be CSV, TSV with the structures in a `Smiles` column
or an SD file. The files may be gzipped.
All entries with failed molecules will be removed.
By default, duplicate entries will be removed by InChIKey (can be turned off with the `--keep_dupl` option)
and structure canonicalization will be performed (can be turned off with the `--nocanon`option).
Omitting structure canonicalization drastically improves the performance.
The output will be a tab-separated text file with SMILES.

Example:
Standardize the ChEMBL SDF download (gzipped), keep only MedChem atoms
and molecules between 3-50 heavy atoms, do not perform canonicalization:
    `$ ./stand_struct.py chembl_29.sdf.gz medchemrac --nocanon`
            

positional arguments:
  in_file               The optionally gzipped input file (CSV, TSV or SDF).
  {full,fullrac,medchem,medchemrac}
                        The output type. 'full': Full dataset, only standardized; 'fullrac': Like 'full', but with stereochemistry removed; 'medchem':
                        Dataset with MedChem filters applied; 'medchemrac': Like 'medchem', but with stereochemistry removed;

optional arguments:
  -h, --help            show this help message and exit
  --nocanon             Turning off structure canonicalization greatly improves performance.
  --min_heavy_atoms MIN_HEAVY_ATOMS
                        The minimum number of heavy atoms for a molecule to be kept (default: 3).
  --max_heavy_atoms MAX_HEAVY_ATOMS
                        The maximum number of heavy atoms for a molecule to be kept (default: 50).
  -d, --keep_duplicates
                        Keep duplicates.
  -c COLUMNS, --columns COLUMNS
                        Comma-separated list of columns to keep (default: all).
```


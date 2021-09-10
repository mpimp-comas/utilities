#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for data calculation.
"""

import gzip
import os.path as op
import subprocess
import tempfile
import uuid
from typing import List, Set, Callable, Union, Tuple

# import sys

import pandas as pd
import numpy as np

from multiprocessing import Pool

try:
    from tqdm.notebook import tqdm

    tqdm.pandas()
    TQDM = True
except ImportError:
    TQDM = False

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Mol
import rdkit.Chem.Descriptors as Desc

# from rdkit.Chem.MolStandardize import rdMolStandardize
# from rdkit.Chem.MolStandardize.validate import Validator
from rdkit.Chem.MolStandardize.charge import Uncharger
from rdkit.Chem.MolStandardize.fragment import LargestFragmentChooser
from rdkit.Chem.MolStandardize.standardize import Standardizer
from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer
from rdkit import rdBase

rdBase.DisableLog("rdApp.info")
# rdBase.DisableLog("rdApp.warn")

molvs_s = Standardizer()
molvs_l = LargestFragmentChooser()
molvs_u = Uncharger()
molvs_t = TautomerCanonicalizer(max_tautomers=100)


def get_value(str_val):
    """convert a string into float or int, if possible."""
    if not str_val:
        return np.nan
    try:
        val = float(str_val)
        if "." not in str_val:
            val = int(val)
    except ValueError:
        val = str_val
    return val


def read_sdf(
    fn, keep_mols=True, merge_prop: str = None, merge_list: Union[List, Set] = None
) -> pd.DataFrame:
    """Create a DataFrame instance from an SD file.
    The input can be a single SD file or a list of files and they can be gzipped (fn ends with `.gz`).
    If a list of files is used, all files need to have the same fields.
    The molecules will be converted to Smiles and can optionally be stored as a `Mol` column.
    Records with no valid molecule will be dropped.

    Parameters:
    ===========
    merge_prop: A property in the SD file on which the file should be merge
        during reading.
    merge_list: A list or set of values on which to merge.
        Only the values of the list are kept.

    Returns:
    ========
    A Pandas DataFrame containing the structures as Smiles.
    """

    d = {"Smiles": []}
    if keep_mols:
        d["Mol"] = []
    ctr = {x: 0 for x in ["In", "Out", "Fail_NoMol"]}
    if merge_prop is not None:
        ctr["NotMerged"] = 0
    first_mol = True
    sd_props = set()
    if not isinstance(fn, list):
        fn = [fn]
    for f in fn:
        do_close = True
        if isinstance(f, str):
            if f.endswith(".gz"):
                file_obj = gzip.open(f, mode="rb")
            else:
                file_obj = open(f, "rb")
        else:
            file_obj = f
            do_close = False
        reader = Chem.ForwardSDMolSupplier(file_obj)
        for mol in reader:
            ctr["In"] += 1
            if not mol:
                ctr["Fail_NoMol"] += 1
                continue
            if first_mol:
                first_mol = False
                # Is the SD file name property used?
                name = mol.GetProp("_Name")
                if len(name) > 0:
                    has_name = True
                    d["Name"] = []
                else:
                    has_name = False
                for prop in mol.GetPropNames():
                    sd_props.add(prop)
                    d[prop] = []
            if merge_prop is not None:
                # Only keep the record when the `merge_prop` value is in `merge_list`:
                if get_value(mol.GetProp(merge_prop)) not in merge_list:
                    ctr["NotMerged"] += 1
                    continue
            mol_props = set()
            ctr["Out"] += 1
            for prop in mol.GetPropNames():
                if prop in sd_props:
                    mol_props.add(prop)
                    d[prop].append(get_value(mol.GetProp(prop)))
                mol.ClearProp(prop)
            if has_name:
                d["Name"].append(get_value(mol.GetProp("_Name")))
                mol.ClearProp("_Name")

            # append NAN to the missing props that were not in the mol:
            missing_props = sd_props - mol_props
            for prop in missing_props:
                d[prop].append(np.nan)
            d["Smiles"].append(mol_to_smiles(mol))
            if keep_mols:
                d["Mol"].append(mol)
        if do_close:
            file_obj.close()
    # Make sure, that all columns have the same length.
    # Although, Pandas would also complain, if this was not the case.
    d_keys = list(d.keys())
    if len(d_keys) > 1:
        k_len = len(d[d_keys[0]])
        for k in d_keys[1:]:
            assert k_len == len(d[k]), f"{k_len=} != {len(d[k])}"
    result = pd.DataFrame(d)
    print(ctr)
    return result


def mol_to_smiles(mol: Mol, canonical: bool = True) -> str:
    """Generate Smiles from mol.

    Parameters:
    ===========
    mol: the input molecule
    canonical: whether to return the canonical Smiles or not

    Returns:
    ========
    The Smiles of the molecule (canonical by default). NAN for failed molecules."""

    if mol is None:
        return np.nan
    try:
        smi = Chem.MolToSmiles(mol, canonical=canonical)
        return smi
    except:
        return np.nan


def smiles_to_mol(smiles: str) -> Mol:
    """Generate a RDKit Molecule from a Smiles.

    Parameters:
    ===========
    smiles: the input string

    Returns:
    ========
    The RDKit Molecule. If the Smiles parsing failed, NAN is returned instead.
    """

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol
        return np.nan
    except:
        return np.nan


def add_mol_col(df: pd.DataFrame, smiles_col="Smiles") -> pd.DataFrame:
    """Add a column containing the RDKit Molecule.

    Parameters:
    ===========
    df: the input DataFrame
    smiles_col: the name of the column containing the Smiles

    Returns:
    ========
    A DataFrame with a column containing the RDKit Molecule.
    """

    df["Mol"] = df[smiles_col].apply(smiles_to_mol)
    return df


def drop_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Remove the list of columns from the dataframe.
    Listed columns that are not available in the dataframe are simply ignored."""
    df = df.copy()
    cols_to_remove = set(cols).intersection(set(df.keys()))
    df = df.drop(cols_to_remove, axis=1)
    return df


def standardize_mol(
    mol,
    largest_fragment=True,
    uncharge=True,
    standardize=True,
    remove_stereo=False,
    canonicalize_tautomer=False,
):
    """Standardize the molecule structures.
    Returns:
    ========
    Smiles of the standardized molecule. NAN for failed molecules."""

    if mol is np.nan or mol is None:
        return np.nan
    if largest_fragment:
        mol = molvs_l.choose(mol)
    if uncharge:
        mol = molvs_u.uncharge(mol)
    if standardize:
        # Apparently, this may fail:
        try:
            mol = molvs_s.standardize(mol)
        except:
            return np.nan
    if remove_stereo:
        mol = molvs_s.stereo_parent(mol)
    if canonicalize_tautomer:
        mol = molvs_t.canonicalize(mol)
    # mol = largest.choose(mol)
    # mol = uncharger.uncharge(mol)
    # mol = normal.normalize(mol)
    # mol = enumerator.Canonicalize(mol)
    return mol_to_smiles(mol)


def standardize_smiles(
    smiles,
    largest_fragment=True,
    uncharge=True,
    standardize=True,
    remove_stereo=False,
    canonicalize_tautomer=False,
) -> str:
    """Creates a molecule from the Smiles string and passes it to `standardize_mol().

    Returns:
    ========
    The Smiles string of the standardized molecule."""
    mol = smiles_to_mol(smiles)  # None handling is done in `standardize_mol`
    result = standardize_mol(
        mol,
        largest_fragment=largest_fragment,
        uncharge=uncharge,
        standardize=standardize,
        remove_stereo=remove_stereo,
        canonicalize_tautomer=canonicalize_tautomer,
    )
    return result


def parallel_pandas(df: pd.DataFrame, func: Callable, workers=6) -> pd.DataFrame:
    """Concurrently apply the `func` to the DataFrame `df`.
    `workers` is the number of parallel threads.
    Currently, TQDM progress bars do not work with the parallel execution.

    Returns:
    ========
    A new Pandas DataFrame.

    Example:
    ========

    >>> def add_props(df):
    >>>     df["Mol"] = df["Smiles"].apply(u.smiles_to_mol)
    >>>     df["LogP"] = df["Mol"].apply(Desc.MolLogP)
    >>>     return df

    >>>     dfs = u.parallel_pandas(df, add_props)
    """
    df = df.copy()
    df_split = np.array_split(df, workers)
    pool = Pool(workers)
    # if TQDM:
    #     result = pd.concat(pool.map(func, df_split))
    # else:
    result = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return result


def get_atom_set(mol):
    result = set()
    for at in mol.GetAtoms():
        result.add(at.GetAtomicNum())
    return result


def filter_mols(
    df: pd.DataFrame, filter: Union[str, List[str]], smiles_col="Smiles", **kwargs
) -> pd.DataFrame:
    """Apply different filters to the molecules.
    If the dataframe contains a `Mol` column, it will be used,
    otherwise the `Mol` column is generated from the `smiles_col` column.
    This might be problematic for large dataframes.
    When in doubt, use `filter_smiles` instead.

    Parameters:
    ===========
    filter [str or list of strings]: The name of the filter to apply.
        Available filters:
            - Isotopes: Keep only non-isotope molecules
            - MedChemAtoms: Keep only molecules with MedChem atoms
            - MinHeavyAtoms: Keep only molecules with 3 or more heavy atoms
            - MaxHeavyAtoms: Keep only molecules with 50 or less heavy atoms
            - Duplicates: Remove duplicates by InChiKey

    kwargs:
        provides the possibility to override the heavy atoms cutoffs:
            - min_heavy_atoms: int
            - max_heavy_atoms: int
    """
    available_filters = {
        "Isotopes",
        "MedChemAtoms",
        "MinHeavyAtoms",
        "MaxHeavyAtoms",
        "Duplicates",
    }
    medchem_atoms = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # 5: Boron
    min_heavy_atoms = kwargs.get("min_heavy_atoms", 3)
    max_heavy_atoms = kwargs.get("max_heavy_atoms", 50)

    def has_non_medchem_atoms(mol):
        if len(get_atom_set(mol) - medchem_atoms) > 0:
            return True
        return False

    def has_isotope(mol) -> bool:
        for at in mol.GetAtoms():
            if at.GetIsotope() != 0:
                return True
        return False

    df = df.copy()
    if isinstance(filter, str):
        filter = [filter]
    for filt in filter:
        if filt not in available_filters:
            raise ValueError(f"Unknown filter: {filt}")
    calc_ha = False
    cols_to_remove = []
    if "Mol" not in df.keys():
        print("Adding molecules...")
        df["Mol"] = df[smiles_col].apply(smiles_to_mol)
        cols_to_remove.append("Mol")
    print(f"Applying filters ({len(filter)})...")
    df = df[~df["Mol"].isnull()]
    for filt in filter:
        if filt == "Isotopes":
            # df = apply_to_smiles(df, smiles_col, {"FiltIsotopes": has_isotope})
            df["FiltIsotopes"] = df["Mol"].apply(has_isotope)
            df = df.query("FiltIsotopes == False")
            cols_to_remove.append("FiltIsotopes")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MedChemAtoms":
            # df = apply_to_smiles(
            #     df, smiles_col, {"FiltNonMCAtoms": has_non_medchem_atoms}
            # )
            df["FiltNonMCAtoms"] = df["Mol"].apply(has_non_medchem_atoms)
            df = df.query("FiltNonMCAtoms == False")
            cols_to_remove.append("FiltNonMCAtoms")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MinHeavyAtoms":
            if not calc_ha:
                # df = apply_to_smiles(
                #     df, smiles_col, {"FiltHeavyAtoms": Desc.HeavyAtomCount}
                # )
                df["FiltHeavyAtoms"] = df["Mol"].apply(Desc.HeavyAtomCount)
                calc_ha = True
            df = df.query(f"FiltHeavyAtoms >= {min_heavy_atoms}")
            cols_to_remove.append("FiltHeavyAtoms")
            print(f"Applied filter {filt} (cutoff: {min_heavy_atoms}): ", end="")
        elif filt == "MaxHeavyAtoms":
            if not calc_ha:
                # df = apply_to_smiles(
                #     df, smiles_col, {"FiltHeavyAtoms": Desc.HeavyAtomCount}
                # )
                df["FiltHeavyAtoms"] = df["Mol"].apply(Desc.HeavyAtomCount)
                calc_ha = True
            df = df.query(f"FiltHeavyAtoms <= {max_heavy_atoms}")
            cols_to_remove.append("FiltHeavyAtoms")
            print(f"Applied filter {filt} (cutoff: {max_heavy_atoms}): ", end="")
        elif filt == "Duplicates":
            # df = apply_to_smiles(
            #     df, smiles_col, {"FiltInChiKey": Chem.inchi.MolToInchiKey}
            # )
            df["FiltInChiKey"] = df["Mol"].apply(Chem.inchi.MolToInchiKey)
            df = df.drop_duplicates(subset="FiltInChiKey")
            cols_to_remove.append("FiltInChiKey")
            print(f"Applied filter {filt}: ", end="")
        else:
            print()
            raise ValueError(f"Unknown filter: {filt}.")
        print(len(df))
    df = drop_cols(df, cols_to_remove)
    return df


def filter_smiles(
    df: pd.DataFrame, filter: Union[str, List[str]], smiles_col="Smiles", **kwargs
) -> pd.DataFrame:
    """Apply different filters to the molecules.
    The molecules are generated from the `smiles_col` column on the fly and are not stored in the DF.
    Make sure, that the DF contains only valid Smiles, first.

    Parameters:
    ===========
    filter [str or list of strings]: The name of the filter to apply.
        Available filters:
            - Isotopes: Keep only non-isotope molecules
            - MedChemAtoms: Keep only molecules with MedChem atoms
            - MinHeavyAtoms: Keep only molecules with 3 or more heavy atoms
            - MaxHeavyAtoms: Keep only molecules with 50 or less heavy atoms
            - Duplicates: Remove duplicates by InChiKey

    kwargs:
        provides the possibility to override the heavy atoms cutoffs:
            - min_heavy_atoms: int
            - max_heavy_atoms: int
    """
    available_filters = {
        "Isotopes",
        "MedChemAtoms",
        "MinHeavyAtoms",
        "MaxHeavyAtoms",
        "Duplicates",
    }
    medchem_atoms = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # 5: Boron
    min_heavy_atoms = kwargs.get("min_heavy_atoms", 3)
    max_heavy_atoms = kwargs.get("max_heavy_atoms", 50)

    def has_non_medchem_atoms(smiles) -> bool:
        mol = smiles_to_mol(smiles)
        if mol is np.nan:
            return True
        if len(get_atom_set(mol) - medchem_atoms) > 0:
            return True
        return False

    def has_isotope(smiles) -> bool:
        mol = smiles_to_mol(smiles)
        if mol is np.nan:
            return True
        for at in mol.GetAtoms():
            if at.GetIsotope() != 0:
                return True
        return False

    def ha(smiles) -> int:
        mol = smiles_to_mol(smiles)
        if mol is np.nan:
            return 0
        return Desc.HeavyAtomCount(mol)

    def inchi(smiles) -> str:
        mol = smiles_to_mol(smiles)
        if mol is np.nan:
            return "NoMol"
        return Chem.inchi.MolToInchiKey(mol)

    df = df.copy()
    if isinstance(filter, str):
        filter = [filter]
    for filt in filter:
        if filt not in available_filters:
            raise ValueError(f"Unknown filter: {filt}")
    calc_ha = False
    cols_to_remove = []
    print(f"Applying filters ({len(filter)})...")
    df = df[~df["Smiles"].isnull()]
    for filt in filter:
        if filt == "Isotopes":
            # df = apply_to_smiles(df, smiles_col, {"FiltIsotopes": has_isotope})
            df["FiltIsotopes"] = df[smiles_col].apply(has_isotope)
            df = df.query("FiltIsotopes == False")
            cols_to_remove.append("FiltIsotopes")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MedChemAtoms":
            # df = apply_to_smiles(
            #     df, smiles_col, {"FiltNonMCAtoms": has_non_medchem_atoms}
            # )
            df["FiltNonMCAtoms"] = df[smiles_col].apply(has_non_medchem_atoms)
            df = df.query("FiltNonMCAtoms == False")
            cols_to_remove.append("FiltNonMCAtoms")
            print(f"Applied filter {filt}: ", end="")
        elif filt == "MinHeavyAtoms":
            if not calc_ha:
                # df = apply_to_smiles(
                #     df, smiles_col, {"FiltHeavyAtoms": Desc.HeavyAtomCount}
                # )
                df["FiltHeavyAtoms"] = df[smiles_col].apply(ha)
                calc_ha = True
            df = df.query(f"FiltHeavyAtoms >= {min_heavy_atoms}")
            cols_to_remove.append("FiltHeavyAtoms")
            print(f"Applied filter {filt} (cutoff: {min_heavy_atoms}): ", end="")
        elif filt == "MaxHeavyAtoms":
            if not calc_ha:
                df["FiltHeavyAtoms"] = df[smiles_col].apply(ha)
                calc_ha = True
            df = df.query(f"FiltHeavyAtoms <= {max_heavy_atoms}")
            cols_to_remove.append("FiltHeavyAtoms")
            print(f"Applied filter {filt} (cutoff: {max_heavy_atoms}): ", end="")
        elif filt == "Duplicates":
            # df = apply_to_smiles(
            #     df, smiles_col, {"FiltInChiKey": Chem.inchi.MolToInchiKey}
            # )
            df["FiltInChiKey"] = df[smiles_col].apply(inchi)
            df = df[~df["FiltInChiKey"] == "NoMol"]
            df = df.drop_duplicates(subset="FiltInChiKey")
            cols_to_remove.append("FiltInChiKey")
            print(f"Applied filter {filt}: ", end="")
        else:
            print()
            raise ValueError(f"Unknown filter: {filt}.")
        print(len(df))
    df = drop_cols(df, cols_to_remove)
    return df


def read_tsv(input_tsv: str) -> pd.DataFrame:
    """Read a tsv file

    Parameters:
    ===========
    input_tsv: Input tsv file

    Returns:
    ========
    The parsed tsv as Pandas DataFrame.
    """
    input_tsv = input_tsv.replace("file://", "")
    df = pd.read_csv(input_tsv, sep="\t")
    return df


def write(text, fn):
    """Write text to a file."""
    with open(fn, "w") as f:
        f.write(text)


def write_tsv(df: pd.DataFrame, output_tsv: str):
    """Write a tsv file, converting the RDKit molecule column to smiles.

    Parameters:
    ===========
    input_tsv: Input tsv file

    """
    # The Mol column can not be saved to TSV in a meaningfull way,
    # so we remove it, if it is present.
    if "Mol" in df.keys():
        df = df.drop("Mol", axis=1)
    df.to_csv(output_tsv, sep="\t", index=False)


def lp(obj, label: str = None, lpad=50, rpad=10):
    """log-printing for different kind of objects"""
    if label is not None:
        label_str = label
    if isinstance(obj, str):
        if label is None:
            label_str = "String"
        print(f"{label_str:{lpad}s}: {obj:>{rpad}s}")
        return

    try:
        shape = obj.shape
        if label is None:
            label_str = "Shape"
        else:
            label_str = f"Shape {label}"
        key_str = ""
        has_nan_str = ""
        try:
            keys = list(obj.columns)
            if len(keys) <= 5:
                key_str = " [ " + ", ".join(keys) + " ] "
            num_nan_cols = ((~obj.notnull()).sum() > 0).sum()
            if num_nan_cols > 0:  # DF has nans
                has_nan_str = f"( NAN values in {num_nan_cols} col(s) )"
        except AttributeError:
            pass
        print(
            f"{label_str:{lpad}s}: {shape[0]:{rpad}d} / {shape[1]:{4}d} {key_str} {has_nan_str}"
        )
        return
    except (TypeError, AttributeError, IndexError):
        pass

    try:
        shape = obj.data.shape
        if label is None:
            label_str = "Shape"
        else:
            label_str = f"Shape {label}"
        key_str = ""
        try:
            keys = list(obj.data.columns)
            if len(keys) <= 5:
                key_str = " [ " + ", ".join(keys) + " ] "
        except AttributeError:
            pass
        num_nan_cols = ((~obj.data.notnull()).sum() > 0).sum()
        has_nan_str = ""
        if num_nan_cols > 0:  # DF has nans
            has_nan_str = f"( NAN values in {num_nan_cols} col(s) )"
        print(
            f"{label_str:{lpad}s}: {shape[0]:{rpad}d} / {shape[1]:{4}d} {key_str} {has_nan_str}"
        )
        return
    except (TypeError, AttributeError, IndexError):
        pass

    try:
        fval = float(obj)
        if label is None:
            label_str = "Number"
        if fval == obj:
            print(f"{label_str:{lpad}s}: {int(obj):{rpad}d}")
        else:
            print(f"{label_str:{lpad}s}: {obj:{rpad+6}.5f}")
        return
    except (ValueError, TypeError):
        # print("Exception")
        pass

    try:
        length = len(obj)
        if label is None:
            label_str = "Length"
        else:
            label_str = f"Length {label}"
        print(f"{label_str:{lpad}s}: {length:{rpad}d}")
        return
    except (TypeError, AttributeError):
        pass

    if label is None:
        label_str = "Object"
    print(f"{label_str:{lpad}s}: {obj}")


def save_list(lst, fn="list.txt"):
    """Save list as text file."""
    with open(fn, "w") as f:
        for line in lst:
            f.write(f"{line}\n")


def load_list(fn="list.txt", as_type=str, skip_remarks=True, skip_empty=True):
    """Read the lines of a text file into a list.

    Parameters:
    ==========
    as_type: Convert the values in the file to the given format. (Default: str).
    skip_remarks: Skip lines starting with `#` (default: True).
    skip_empty: Skip empty lines. (Default: True).

    Returns:
    ========
    A list of values of the given type.
    """
    result = []
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if skip_empty and len(line) == 0:
                continue
            if skip_remarks and line.startswith("#"):
                continue
            result.append(as_type(line))
    return result


def open_in_localc(df: pd.DataFrame):
    """Open a Pandas DataFrame in LO Calc for visual inspection."""
    td = tempfile.gettempdir()
    tf = str(uuid.uuid4()).split("-")[0] + ".tsv"
    path = op.join(td, tf)
    write_tsv(df, path)
    subprocess.Popen(["localc", path])


def listify(s, sep=" ", as_int=True, strip=True):
    """A helper func for the Jupyter Notebook,
    which generates a correctly formatted list out of pasted text."""
    to_number = int if as_int else float
    result = []
    if s.startswith("["):
        s = s[1:]
    if s.endswith("]"):
        s = s[:-1]
    lst = s.split(sep)
    for el in lst:
        if strip:
            el = el.strip()
        if len(el) == 0:
            continue
        try:
            el = to_number(el)
        except ValueError:
            pass
        result.append(el)
    return result


def filter(
    df: pd.DataFrame, mask, reset_index=True, print_len=True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filters a dataframe and returns the passing fraction and the failing fraction as
    two separate dataframes.

    Returns: passing and failing dataframe."""
    df_pass = df[mask].copy()
    df_fail = df[~mask].copy()
    if reset_index:
        df_pass = df_pass.reset_index(drop=True)
        df_fail = df_fail.reset_index(drop=True)
    if print_len:
        print(f"Pass: {len(df_pass):8d}  Fail: {len(df_fail):8d}")
    return df_pass, df_fail


def groupby(df_in, by=None, num_agg=["median", "mad", "count"], str_agg="unique"):
    """Other str_aggs: "first", "unique"."""

    def _concat(values):
        return "; ".join(str(x) for x in values)

    def _unique(values):
        return "; ".join(set(str(x) for x in values))

    if isinstance(num_agg, str):
        num_agg = [num_agg]
    df_keys = df_in.columns
    numeric_cols = list(df_in.select_dtypes(include=[np.number]).columns)
    str_cols = list(set(df_keys) - set(numeric_cols))
    # if by in numeric_cols:
    try:
        by_pos = numeric_cols.index(by)
        numeric_cols.pop(by_pos)
    except ValueError:
        pass
    try:
        by_pos = str_cols.index(by)
        str_cols.pop(by_pos)
    except ValueError:
        pass
    aggregation = {}
    for k in numeric_cols:
        aggregation[k] = num_agg
    if str_agg == "join":
        str_agg_method = _concat
    elif str_agg == "first":
        str_agg_method = "first"
    elif str_agg == "unique":
        str_agg_method = _unique
    for k in str_cols:
        aggregation[k] = str_agg_method
    df = df_in.groupby(by)
    df = df.agg(aggregation).reset_index()
    df_cols = [
        "_".join(col).strip("_").replace("_<lambda>", "").replace("__unique", "")
        for col in df.columns.values
    ]
    df.columns = df_cols
    return df

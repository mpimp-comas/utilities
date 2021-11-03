#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###########################
Standardize Structure Files
###########################

*Created on Tue Aug 31 2021 08:45  by A. Pahl*

Standardize and filter SD files, e.g. the ChEMBL dataset."""

import sys
import gzip
import csv
from copy import deepcopy
import argparse
import signal
from contextlib import contextmanager

from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Mol
import rdkit.Chem.Descriptors as Desc
from rdkit.Chem.Scaffolds import MurckoScaffold

from rdkit.Chem.MolStandardize.charge import Uncharger
from rdkit.Chem.MolStandardize.fragment import LargestFragmentChooser
from rdkit.Chem.MolStandardize.standardize import Standardizer
from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer

from rdkit import RDLogger

LOG = RDLogger.logger()
LOG.setLevel(RDLogger.CRITICAL)


# Timeout code is taken from José's NPFC project:
# https://github.com/mpimp-comas/npfc/blob/master/npfc/utils.py
def raise_timeout(signum, frame):
    """Function to actually raise the TimeoutError when the time has come."""
    raise TimeoutError


@contextmanager
def timeout(time):
    # register a function to raise a TimeoutError on the signal.
    signal.signal(signal.SIGALRM, raise_timeout)
    # schedule the signal to be sent after time
    signal.alarm(time)
    # run the code block within the with statement
    try:
        yield
    except TimeoutError:
        pass  # exit the with statement
    finally:
        # unregister the signal so it won't be triggered if the timeout is not reached
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def get_value(str_val):
    """convert a string into float or int, if possible."""
    if not str_val:
        return ""
    if str_val is None:
        return ""
    try:
        val = float(str_val)
        if "." not in str_val:
            val = int(val)
    except ValueError:
        val = str_val
    return val


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
        return None
    try:
        smi = Chem.MolToSmiles(mol, canonical=canonical)
        return smi
    except:
        return None


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
        return None
    except:
        return None


def get_atom_set(mol):
    result = set()
    for at in mol.GetAtoms():
        result.add(at.GetAtomicNum())
    return result


def has_isotope(mol: Mol) -> bool:
    for at in mol.GetAtoms():
        if at.GetIsotope() != 0:
            return True
    return False


def csv_supplier(fo, dialect):
    reader = csv.DictReader(fo, dialect=dialect)
    for row in reader:
        mol = smiles_to_mol(row["Smiles"])
        if mol is None:
            yield {"Mol": None}
            continue
        d = {}
        for prop in row:
            if prop == "Smiles":
                continue
            d[prop] = get_value(row[prop])
        d["Mol"] = mol
        yield d


def sdf_supplier(fo):
    reader = Chem.ForwardSDMolSupplier(fo)
    for mol in reader:
        if mol is None:
            yield {"Mol": None}
            continue
        d = {}
        # Is the SD file name property used?
        name = mol.GetProp("_Name")
        if len(name) > 0:
            d["Name"] = get_value(name)
        for prop in mol.GetPropNames():
            d[prop] = get_value(mol.GetProp(prop))

        for prop in mol.GetPropNames():
            d[prop] = get_value(mol.GetProp(prop))
            mol.ClearProp(prop)
        d["Mol"] = mol
        yield d


def process(
    fn: str,
    out_type: str,
    canon: bool,
    columns: str,  # comma separated list of columns to keep
    min_heavy_atoms: int,
    max_heavy_atoms: int,
    keep_dupl: bool,
    verbose: bool,
    every_n: int,
):
    medchem_atoms = {1, 5, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # 5: Boron
    molvs_s = Standardizer()
    molvs_l = LargestFragmentChooser()
    molvs_u = Uncharger()
    molvs_t = TautomerCanonicalizer(max_tautomers=100)

    canon_str = ""
    if not canon:
        canon_str = "_nocanon"
    dupl_str = ""
    if keep_dupl:
        dupl_str = "_dupl"
    min_ha_str = ""
    max_ha_str = ""
    if "medchem" in out_type:
        if min_heavy_atoms != 3:
            min_ha_str = f"_minha_{min_heavy_atoms}"
        if max_heavy_atoms != 50:
            max_ha_str = f"_maxha_{min_heavy_atoms}"

    if len(columns) > 0:
        columns = set(columns.split(","))
    else:
        columns = set()
    header = []
    ctr = {x: 0 for x in ["In", "Out", "Fail_NoMol", "Duplicates", "Filter", "Timeout"]}
    first_mol = True
    sd_props = set()
    inchi_keys = set()
    fn = fn.split(",")  # allow comma separated list of files
    first_dot = fn[0].find(".")
    fn_base = fn[0][:first_dot]
    out_fn = f"{fn_base}_{out_type}{canon_str}{dupl_str}{min_ha_str}{max_ha_str}.tsv"
    outfile = open(out_fn, "w")
    # Initialize reader for the correct input type

    if verbose:
        # Add file name info and print newline after each info line.
        fn_info = f"({fn_base})"
        end_char = "\n"
    else:
        fn_info = ""
        end_char = "\r"

    for f in fn:
        do_close = True
        if "sd" in f:
            if f.endswith(".gz"):
                file_obj = gzip.open(f, mode="rb")
            else:
                file_obj = open(f, "rb")
            reader = sdf_supplier(file_obj)
        elif "csv" in f:
            if f.endswith(".gz"):
                file_obj = gzip.open(f, mode="rb")
            else:
                file_obj = open(f, "r")
            reader = csv_supplier(file_obj, dialect="excel")
        elif "tsv" in f:
            if f.endswith(".gz"):
                file_obj = gzip.open(f, mode="rb")
            else:
                file_obj = open(f, "r")
            reader = csv_supplier(file_obj, dialect="excel-tab")
        else:
            raise ValueError(f"Unknown input file format: {f}")

        for rec in reader:
            ctr["In"] += 1
            mol = rec["Mol"]
            if mol is None:
                ctr["Fail_NoMol"] += 1
                continue
            if first_mol:
                first_mol = False
                header = [x for x in rec if x != "Mol"]
                if len(columns) > 0:
                    header = [x for x in header if x in columns]
                header.append("InChIKey")
                sd_props = set(header.copy())
                header.append("Smiles")
                outfile.write("\t".join(header) + "\n")

            mol_props = set()
            d = {}
            for prop in rec:
                if prop in sd_props:
                    if prop == "Mol":
                        continue
                    mol_props.add(prop)
                    d[prop] = rec[prop]

            # append "" to the missing props that were not in the mol:
            missing_props = sd_props - mol_props
            for prop in missing_props:
                d[prop] = ""

            # Standardization
            mol = molvs_l.choose(mol)
            mol = molvs_u.uncharge(mol)
            # Apparently, this may fail:
            try:
                mol = molvs_s.standardize(mol)
            except:
                ctr["Fail_NoMol"] += 1
                continue

            # "murcko" implies "rac"
            if "rac" in out_type or "murcko" in out_type:
                mol = molvs_s.stereo_parent(mol)
            if mol is None:
                ctr["Fail_NoMol"] += 1
                continue

            if "murcko" in out_type:
                mol = MurckoScaffold.GetScaffoldForMol(mol)
                if mol is None:
                    ctr["Fail_NoMol"] += 1
                    continue

            if not canon:
                # When canonicalization is not performed,
                # we can check for duplicates already here:
                try:
                    inchi = Chem.inchi.MolToInchiKey(mol)
                except:
                    ctr["Fail_NoMol"] += 1
                    continue
                if not keep_dupl:
                    if inchi in inchi_keys:
                        ctr["Duplicates"] += 1
                        continue
                    inchi_keys.add(inchi)
                d["InChIKey"] = inchi

            # MedChem filters:
            if "medchem" in out_type:
                # Only MedChem atoms:
                if len(get_atom_set(mol) - medchem_atoms) > 0:
                    ctr["Filter"] += 1
                    continue
                # No isotopes:
                if has_isotope(mol):
                    ctr["Filter"] += 1
                    continue
                # HeavyAtom >= 3 or <= 50:
                ha = Desc.HeavyAtomCount(mol)
                if ha < min_heavy_atoms or ha > max_heavy_atoms:
                    ctr["Filter"] += 1
                    continue

            if canon:
                # Late canonicalization, because it is so expensive:
                mol_copy = deepcopy(mol)  # copy the mol to restore it after a timeout
                timed_out = True
                with timeout(2):
                    try:
                        mol = molvs_t.canonicalize(mol)
                    except:
                        # in case of a canonicalization error, restore original mol
                        mol = mol_copy
                    timed_out = False
                if timed_out:
                    ctr[
                        "Timeout"
                    ] += 1  # increase the counter but do not fail the entry
                    mol = mol_copy  # instead, restore from the copy
                if mol is None:
                    ctr["Fail_NoMol"] += 1
                    continue
                try:
                    inchi = Chem.inchi.MolToInchiKey(mol)
                except:
                    ctr["Fail_NoMol"] += 1
                    continue
                if not keep_dupl:
                    # When canonicalization IS performed,
                    # we have to check for duplicates now:
                    if inchi in inchi_keys:
                        ctr["Duplicates"] += 1
                        continue
                    inchi_keys.add(inchi)
                d["InChIKey"] = inchi

            smi = mol_to_smiles(mol)
            if smi is None:
                ctr["Fail_NoMol"] += 1
                continue
            d["Smiles"] = smi
            ctr["Out"] += 1
            line = [str(d[x]) for x in header]
            outfile.write("\t".join(line) + "\n")

            if ctr["In"] % every_n == 0:
                print(
                    f"{fn_info}  In: {ctr['In']:8d}  Out: {ctr['Out']: 8d}  Failed: {ctr['Fail_NoMol']:6d}  "
                    f"Dupl: {ctr['Duplicates']:6d}  Filt: {ctr['Filter']:6d}  Timeout: {ctr['Timeout']:6d}       ",
                    end=end_char,
                )
                sys.stdout.flush()

        if do_close:
            file_obj.close()
    outfile.close()
    print(
        f"{fn_info}  In: {ctr['In']:8d}  Out: {ctr['Out']: 8d}  Failed: {ctr['Fail_NoMol']:6d}  "
        f"Dupl: {ctr['Duplicates']:6d}  Filt: {ctr['Filter']:6d}  Timeout: {ctr['Timeout']:6d}   done."
    )
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Standardize structures. Input files can be CSV, TSV with the structures in a `Smiles` column
or an SD file. The files may be gzipped.
All entries with failed molecules will be removed.
By default, duplicate entries will be removed by InChIKey (can be turned off with the `--keep_dupl` option)
and structure canonicalization will be performed (can be turned off with the `--nocanon`option),
where a timeout is enforced on the canonicalization if it takes longer than 2 seconds per structure.
Timed-out structures WILL NOT BE REMOVED, they are kept in their state before canonicalization.
Omitting structure canonicalization drastically improves the performance.
The output will be a tab-separated text file with SMILES.

Example:
Standardize the ChEMBL SDF download (gzipped), keep only MedChem atoms
and molecules between 3-50 heavy atoms, do not perform canonicalization:
    `$ ./stand_struct.py chembl_29.sdf.gz medchemrac --nocanon`
            """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "in_file",
        help="The optionally gzipped input file (CSV, TSV or SDF). Can also be a comma-separated list of file names.",
    )
    parser.add_argument(
        "output_type",
        choices=[
            "full",
            "fullrac",
            "medchem",
            "medchemrac",
            "fullmurcko",
            "medchemmurcko",
        ],
        help=(
            "The output type. "
            "'full': Full dataset, only standardized; "
            "'fullrac': Like 'full', but with stereochemistry removed; "
            "'fullmurcko': Like 'fullrac', structures are reduced to their Murcko scaffolds; "
            "'medchem': Dataset with MedChem filters applied, bounds for the number of heavy atoms can be optionally given; "
            "'medchemrac': Like 'medchem', but with stereochemistry removed; "
            "'medchemmurcko': Like 'medchemrac', structures are reduced to their Murcko scaffolds; "
            "(all filters, canonicalization and duplicate checks are applied after Murcko generation)."
        ),
    )
    parser.add_argument(
        "--nocanon",
        action="store_true",
        help="Turning off structure canonicalization greatly improves performance.",
    )
    parser.add_argument(
        "--min_heavy_atoms",
        type=int,
        default=3,
        help="The minimum number of heavy atoms for a molecule to be kept (default: 3).",
    )
    parser.add_argument(
        "--max_heavy_atoms",
        type=int,
        default=50,
        help="The maximum number of heavy atoms for a molecule to be kept (default: 50).",
    )
    parser.add_argument(
        "-d", "--keep_duplicates", action="store_true", help="Keep duplicates."
    )
    parser.add_argument(
        "-c",
        "--columns",
        type=str,
        default="",
        help="Comma-separated list of columns to keep (default: all).",
    )
    parser.add_argument(
        "-n",
        type=int,
        default=1000,
        help="Show info every `N` records (default: 1000).",
    )
    parser.add_argument(
        "-v",
        action="store_true",
        help="Turn on verbose status output.",
    )
    args = parser.parse_args()
    print(args)
    process(
        args.in_file,
        args.output_type,
        not args.nocanon,
        args.columns,
        args.min_heavy_atoms,
        args.max_heavy_atoms,
        args.keep_duplicates,
        args.v,
        args.n,
    )

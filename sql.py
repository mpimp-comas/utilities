"""Convenience functions for working with chemistry-containing (Smiles)
SQLite databases."""

import sqlite3

from rdkit.Chem import AllChem as Chem

# COMAS-specific code
PATH = "/home/pahl/comas/share"


def sss(smiles: str, query: str):
    """
    An extension for SQLite to enable basic substructure search with Smiles.
    Enable this extension in the connected DB by:
    `conn.create_function("sss", 2, sql.sss)`

    Complete example query:
    ``` Python
    conn = sqlite3.connect(f"{PATH_TO_SQLITE_DB}")
    conn.create_function("sss", 2, sql.sss)
    smi = 'c2ccc1[nH]ccc1c2'  # indole
    df = pd.read_sql(f"select Compound_Id, Smiles, sss(Smiles, '{q}') as Found from Compound where Found > 0;", conn)
    ```"""
    try:
        m = Chem.MolFromSmiles(smiles)
    except:
        return 0
    if m is None:
        return 0
    q = Chem.MolFromSmiles(query)

    if m.HasSubstructMatch(q):
        return 1
    return 0


def connect_to_comas_db() -> sqlite3.Connection:
    """COMAS-specific convenience code.
    Connect to an SQLite DB created from some export files
    and enable the SSS extension."""
    conn = sqlite3.connect(f"{PATH}/comas.sqlite")
    conn.create_function("sss", 2, sss)
    return conn

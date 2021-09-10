#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
########################################
Create SQLite DB from COMAS export files
########################################

*Created on THu Sep 9 2021 09:00  by A. Pahl*

Create an SQLite DB from the COMAS DB export files."""

import sqlite3
import pandas as pd


if __name__ == "__main__":
    PATH = "/home/pahl/comas/share"
    conn = sqlite3.connect(f"{PATH}/comas.sqlite")

    df = pd.read_csv(f"{PATH}/comas_smiles.tsv", sep="\t")
    df.to_sql("Compound", conn, if_exists="replace", index=False)

    df = pd.read_csv(f"{PATH}/comas_batch_data.tsv", sep="\t")
    df.to_sql("Batch", conn, if_exists="replace", index=False)

    df = pd.read_csv(f"{PATH}/comas_container.tsv", sep="\t")
    df.to_sql("Container", conn, if_exists="replace", index=False)

    conn.commit()
    conn.close()

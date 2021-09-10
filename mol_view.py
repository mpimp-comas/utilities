#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for viewing DataFrames with molecules.
"""
# import os.path as op

import base64
import time
from itertools import chain
from io import BytesIO as IO


import pandas as pd
import numpy as np

pd.set_option("display.max_colwidth", None)

from PIL import Image, ImageChops

from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdCoordGen import AddCoords  # New coord. generation
from rdkit.Chem import Draw

import utils, templ

USE_RDKIT_NEW_COORD = True
BGCOLOR = "#94CAEF"
IMG_GRID_SIZE = 235


def is_interactive_ipython():
    try:
        get_ipython()  # type: ignore
        ipy = True
        # print("> interactive IPython session.")
    except NameError:
        ipy = False
    return ipy


IPYTHON = is_interactive_ipython()
if IPYTHON:
    from IPython.core.display import HTML


def rescale(mol, f=1.4):
    tm = np.zeros((4, 4), np.double)
    for i in range(3):
        tm[i, i] = f
    tm[3, 3] = 1.0
    Chem.TransformMol(mol, tm)


def add_coords(mol, force=False):
    """Check if a mol has 2D coordinates and if not, calculate them."""
    if not force:
        try:
            mol.GetConformer()
        except ValueError:
            force = True  # no 2D coords... calculate them

    if force:
        if USE_RDKIT_NEW_COORD and mol.GetNumAtoms() <= 75:
            AddCoords(mol)
            rescale(mol, f=1.4)
        else:
            mol.Compute2DCoords()


def make_transparent(img):
    img = img.convert("RGBA")
    pixdata = img.load()
    width, height = img.size
    for y in range(height):
        for x in range(width):
            if pixdata[x, y] == (255, 255, 255, 255):
                pixdata[x, y] = (255, 255, 255, 0)
    return img


def autocrop(im, bgcolor="white"):
    if im.mode != "RGB":
        im = im.convert("RGB")
    bg = Image.new("RGB", im.size, bgcolor)
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return None  # no contents


def mol_img_file(mol, size=300, hlsss=None):
    img_file = IO()
    if isinstance(mol, list):
        [add_coords(x) for x in mol]
        img = autocrop(Draw.MolsToGridImage(mol, size=(size, size)))
    else:
        if isinstance(mol, str):  # convert from Smiles on-the-fly, when necessary
            if len(mol) > 0:
                mol = Chem.MolFromSmiles(mol)
            else:
                mol = Chem.MolFromSmiles("*")
        if mol is None or mol is np.nan:
            mol = Chem.MolFromSmiles("*")
        if hlsss is not None:
            if isinstance(hlsss, str):
                hlsss = hlsss.split(",")
                atoms = set()
                for smi in hlsss:
                    m = Chem.MolFromSmiles(smi)
                    if m:
                        matches = list(chain(*mol.GetSubstructMatches(m)))
                    else:
                        matches = []
                    if len(matches) > 0:
                        atoms = atoms.union(set(matches))
            atoms = list(atoms)
        else:
            atoms = []
        try:
            add_coords(mol)
            img = autocrop(
                Draw.MolToImage(mol, size=(size, size), highlightAtoms=atoms)
            )
        except UnicodeEncodeError:
            print(Chem.MolToSmiles(mol))
            mol = Chem.MolFromSmiles("*")
            img = autocrop(Draw.MolToImage(mol, size=(size, size)))
    img = make_transparent(img)
    img.save(img_file, format="PNG")
    val = img_file.getvalue()
    img_file.close()
    return val


def b64_mol(mol, size=300, hlsss=None):
    img_file = mol_img_file(mol, size=size, hlsss=hlsss)
    b64 = base64.b64encode(img_file)
    b64 = b64.decode()
    return b64


def mol_img_tag(mol, size=300, options=None):
    tag = """<img {} src="data:image/png;base64,{}" alt="Mol"/>"""
    if options is None:
        options = ""
    img_tag = tag.format(options, b64_mol(mol, size=size))
    return img_tag


def write(text, fn):
    with open(fn, "w") as f:
        f.write(text)


def _mol_img_tag(mol):
    return pd.Series(mol_img_tag(mol))


def _apply_link(input, link, ln_title="Link"):
    """input[0]: mol_img_tag
    input[1]: link_col value"""
    link_str = link.format(input[1])
    result = '<a target="_blank" href="{}" title="{}">{}</a>'.format(
        link_str, ln_title, input[0]
    )
    return result


def show_mols(mols_or_smiles, cols=5):
    """A small utility to quickly view a list of mols (or Smiles) in a grid."""
    html_list = ['<table align="center">']
    idx = 0
    row_line = []
    if not isinstance(mols_or_smiles, list):
        mols_or_smiles = [mols_or_smiles]
    for mol in mols_or_smiles:
        idx += 1
        img = mol_img_tag(mol)
        cell = "<td>{}<td>".format(img)
        row_line.append(cell)
        if idx == cols:
            row = "<tr>" + "".join(row_line) + "</td>"
            html_list.append(row)
            idx = 0
            row_line = []
    if idx != 0:
        row = "<tr>" + "".join(row_line) + "</td>"
        html_list.append(row)
    html_list.append("</table>")
    table = "\n".join(html_list)
    if IPYTHON:
        return HTML(table)
    return table


def mol_grid(
    df,
    drop=[],
    keep=[],
    hide=[],
    smiles_col="Smiles",
    mol_col="Mol",
    id_col="Compound_Id",
    size=IMG_GRID_SIZE,
    as_html=True,
    **kwargs,
):
    """Creates a HTML grid out of the DataFrame input.

    Parameters:
        df (DataFrame): Pandas DataFrame with either a Smiles column or a Mol column.
        interactive (bool)
        as_html: return a Jupyter HTML object, when possible, otherwise return a string.
        link_templ, link_col (str) (then interactive is false)
    Returns:
        HTML table as TEXT with molecules in grid-like layout to embed in IPython or a web page."""

    interact = kwargs.get("interactive", False)
    link_templ = kwargs.get("link_templ", None)
    link_col = kwargs.get("link_col", None)
    if link_col is not None:
        interact = (
            False  # interact is not avail. when clicking the image should open a link
        )
    mols_per_row = kwargs.get("mols_per_row", 5)
    hlsss = kwargs.get(
        "hlsss", None
    )  # colname with Smiles (,-separated) for Atom highlighting
    truncate = kwargs.get("truncate", 12)
    img_dir = None
    df = df.copy()
    if mol_col not in df.keys():
        df = utils.add_mol_col(df, smiles_col=smiles_col)
    if len(keep) > 0:
        keep.append(mol_col)
        if id_col is not None and id_col not in keep:
            keep.append(id_col)
        df = df[keep]
    drop.append(smiles_col)
    df = utils.drop_cols(df, drop)
    props = []
    for x in list(df.keys()):
        if x != mol_col and x != id_col and x != hlsss and (x not in hide):
            props.append(x)
    time_stamp = time.strftime("%y%m%d%H%M%S")
    td_opt = {"style": "text-align: center;"}
    header_opt = {"bgcolor": BGCOLOR}
    table_list = []
    guessed_id = id_col
    if guessed_id not in df.keys():
        guessed_id = None
    if interact and guessed_id is not None:
        table_list.append(templ.TBL_JAVASCRIPT.format(ts=time_stamp, bgcolor=BGCOLOR))

    if len(props) > 0:
        td_opt["colspan"] = "2"
        prop_row_cells = {k: [] for k, _ in enumerate(props)}

    rows = []
    id_cells = []
    mol_cells = []
    for idx, (_, rec) in enumerate(df.iterrows(), 1):
        mol = rec[mol_col]
        if guessed_id:
            id_prop_val = str(rec[guessed_id])
            img_id = id_prop_val
            cell_opt = {"id": "{}_{}".format(id_prop_val, time_stamp)}
            cell_opt.update(td_opt)
            cell_opt.update(header_opt)
            id_cells.extend(templ.td(id_prop_val, cell_opt))
        else:
            img_id = idx

        if not mol:
            cell = ["no structure"]

        else:
            if hlsss is not None:
                hlsss_smi = rec[hlsss]
            else:
                hlsss_smi = None
            if img_dir is None:  # embed the images in the doc
                img_src = b64_mol(mol, size * 2, hlsss=hlsss_smi)

            if interact and guessed_id is not None:
                img_opt = {
                    "title": "Click to select / unselect",
                    "onclick": "toggleCpd('{}')".format(id_prop_val),
                }
            elif link_col is not None:

                img_opt = {"title": "Click to open link"}
                #          "onclick": "location.href='{}';".format(link)}
                # '<a target="_blank" href="{}" title="{}">{}</a>'

            else:
                img_opt = {"title": str(img_id)}

            img_opt["style"] = f"max-width: {size}px; max-height: {size}px;"
            # img_opt["height"] = "{}px".format(size)
            cell = templ.img(img_src, img_opt)
            if link_col is not None:
                link = link_templ.format(rec[link_col])
                a_opt = {"href": link}
                cell = templ.a(cell, a_opt)

        # td_opt = {"align": "center"}
        td_opt = {
            "style": "text-align: center;",
            "bgcolor": "#FFFFFF",
        }
        if len(props) > 0:
            td_opt["colspan"] = "2"

        mol_cells.extend(templ.td(cell, td_opt))

        if len(props) > 0:
            for prop_no, prop in enumerate(props):
                prop_opt = {"style": "text-align: left;"}
                val_opt = {"style": "text-align: left;"}
                prop_cells = []
                prop_val = ""
                if prop in rec:
                    prop_val = str(rec[prop])
                    if (
                        prop == "Pure_Flag"
                        and prop_val != ""
                        and prop_val != "n.d."
                        and "Purity" in rec
                        and "LCMS_Date" in rec
                    ):
                        val_opt["title"] = "{}% ({})".format(
                            rec["Purity"], rec["LCMS_Date"]
                        )
                prop_cells.extend(templ.td(prop[:25], prop_opt))
                prop_cells.extend(
                    templ.td(
                        templ.div(
                            prop_val[:truncate],
                            options={"style": f"max-width: {size-20}px;"},
                        ),
                        val_opt,
                    ),
                )
                prop_row_cells[prop_no].extend(prop_cells)

        if idx % mols_per_row == 0 or idx == len(df):
            if guessed_id:
                rows.extend(templ.tr(id_cells))
            rows.extend(templ.tr(mol_cells))

            if len(props) > 0:
                colspan_factor = 2
                for prop_no in sorted(prop_row_cells):
                    rows.extend(templ.tr(prop_row_cells[prop_no]))
                prop_row_cells = {k: [] for k, _ in enumerate(props)}
            else:
                colspan_factor = 1
            empty_row_options = {"colspan": mols_per_row * colspan_factor}
            empty_row_options["style"] = "border: none;"
            empty_row = templ.tr(templ.td("&nbsp;", options=empty_row_options))
            rows.extend(empty_row)
            id_cells = []
            mol_cells = []

    table_list.extend(templ.table(rows))

    if interact and guessed_id is not None:
        table_list.append(templ.ID_LIST.format(ts=time_stamp))

    # print(table_list)
    if IPYTHON and as_html:
        return HTML("".join(table_list))
    return "".join(table_list)


def write_mol_grid(
    df,
    title="MolGrid",
    fn="molgrid.html",
    drop=[],
    keep=[],
    smiles_col="Smiles",
    mol_col="Mol",
    id_col="Compound_Id",
    **kwargs,
):
    """
    Write a grid of molecules to a file and return the link.

    Args:
        df: DataFrame with molecules.
        title: Document title.
        fn: Filename to write.
    """
    tbl = mol_grid(
        df,
        title=title,
        drop=drop,
        keep=keep,
        smiles_col=smiles_col,
        mol_col=mol_col,
        id_col=id_col,
        size=IMG_GRID_SIZE,
        as_html=False,
        **kwargs,
    )
    header = kwargs.get("header", None)
    summary = kwargs.get("summary", None)
    page = templ.page(tbl, title=title, header=header, summary=summary)
    utils.write(page, fn)
    if IPYTHON:
        return HTML('<a href="{}">{}</a>'.format(fn, title))

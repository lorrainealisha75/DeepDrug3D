from biopandas.pdb import PandasPdb


def get_binding_site_centre(complex, ligand_name):
    hetatm = complex.df['HETATM']
    ligand = hetatm.loc[hetatm['residue_name'] == ligand_name]
    x = ligand['x_coord']
    y = ligand['y_coord']
    z = ligand['z_coord']
    return round(x.mean(), 2), round(y.mean(), 2), round(z.mean(), 2)


def get_distances(complex, binding_site_coord):
    return complex.distance(xyz=binding_site_coord, records=('ATOM'))


def get_binding_residue_ids(complex, distances, radius):
    atoms = complex.df['ATOM'][distances < radius]
    atom_list = atoms.loc[atoms['element_symbol'] != 'H']['residue_number'].drop_duplicates()
    return atom_list.tolist()


def get_ligand_residue_id(pl_complex, ligand_name):
    hetatm = pl_complex.df['HETATM']
    ligand = hetatm.loc[hetatm['residue_name'] == ligand_name]
    residue_id = ligand['residue_number']
    return residue_id.drop_duplicates().tolist()


def create_aux_file(ligand, ids, bind_site_centre):
    f = open(ligand + '_aux.txt', 'w')
    f.write('BindingResidueIDs:')
    for id in ids:
        f.write(str(id)+' ')
    f.write('\n')
    f.write('BindingSiteCenter:')
    for coord in bind_site_centre:
        f.write(str(coord)+' ')
    f.close()


def main():
    pdb_file_path = "data/2yki.pdb"
    ligand_name = "YKI"
    r = 6.0

    pl_complex = PandasPdb().read_pdb(pdb_file_path)
    x, y, z = get_binding_site_centre(pl_complex, ligand_name)

    distances = get_distances(pl_complex, (x, y, z))

    binding_residue_ids = get_binding_residue_ids(pl_complex, distances, r)

    ligand_id = get_ligand_residue_id(pl_complex, ligand_name)

    all_ids = binding_residue_ids + ligand_id

    create_aux_file(ligand_name, all_ids, (x, y, z))


if __name__ == "__main__":
    main()
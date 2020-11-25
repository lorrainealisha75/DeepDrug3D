#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script converts the pdb file to the voxel representation.
@author: Limeng Pu
"""
from __future__ import division

import sys
import os
import json
import argparse
import numpy as np

import pybel
from biopandas.pdb import PandasPdb

from binding_grid import Grid3DBuilder


# from keras.models import load_model

def site_voxelization(site, voxel_length):
    site = np.array(site, dtype=np.float64)
    coords = site[:, 0:3]
    potentials = site[:, 3:None]
    voxel_length = 32
    voxel_start = int(-voxel_length / 2 + 1)
    voxel_end = int(voxel_length / 2)
    voxel = np.zeros(shape=(14, voxel_length, voxel_length, voxel_length),
                     dtype=np.float64)
    cnt = 0
    for x in xrange(voxel_start, voxel_end + 1, 1):
        for y in xrange(voxel_start, voxel_end + 1, 1):
            for z in xrange(voxel_start, voxel_end + 1, 1):
                temp_voxloc = [x, y, z]
                distances = np.linalg.norm(coords - temp_voxloc, axis=1)
                min_dist = np.min(distances)
                index = np.where(distances == min_dist)
                if min_dist < 0.01:
                    voxel[:, x - voxel_start, y - voxel_start, z - voxel_start] = potentials[index, :]
                    cnt += 1
                else:
                    voxel[:, x - voxel_start, y - voxel_start, z - voxel_start] = np.ones((14,))
    return voxel


def normalize(v):
    """ vector normalization """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def vrrotvec(a, b):
    """ Function to rotate one vector to another, inspired by
    vrrotvec.m in MATLAB """
    a = normalize(a)
    b = normalize(b)
    ax = normalize(np.cross(a, b))
    angle = np.arccos(np.minimum(np.dot(a, b), [1]))
    if not np.any(ax):
        absa = np.abs(a)
        mind = np.argmin(absa)
        c = np.zeros((1, 3))
        c[mind] = 0
        ax = normalize(np.cross(a, c))
    r = np.concatenate((ax, angle))
    return r


def vrrotvec2mat(r):
    """ Convert the axis-angle representation to the matrix representation of the
    rotation """
    s = np.sin(r[3])
    c = np.cos(r[3])
    t = 1 - c

    n = normalize(r[0:3])

    x = n[0]
    y = n[1]
    z = n[2]

    m = np.array(
        [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
         [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
         [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]
    )
    return m


class Vox3DBuilder(object):
    """
    This class convert the pdb file to the voxel representation for the input
    of deep learning architecture. The conversion is around 30 mins for each binding site.
    """

    @staticmethod
    def voxelization(pdb_path, aux_input_path, r, N):
        # Read the pdb file
        ppdb = PandasPdb().read_pdb(pdb_path)
        protein_all_atoms_df = ppdb.df['ATOM']  # dataframe with list of protein atoms
        # unlike in the original implementation we want to exclude H atoms
        protein_df = protein_all_atoms_df[protein_all_atoms_df['atom_name'].str[0] != 'H']
        content = []

        # Read the aux file
        with open(aux_input_path) as json_file:
            data = json.load(json_file)
            content.append(data["residue_ids"])
            content.append(data["binding_site_coords"])
        json_file.close()
        resi = content[0]
        if len(content[1]) != 0:
            pocket_df = protein_df[protein_df['residue_number'].isin(resi)]
            pocket_coords = np.array([pocket_df['x_coord'], pocket_df['y_coord'], pocket_df['z_coord']]).T
            pocket_center = list(content[1].values())
        else:
            print('No center is provided')
            pocket_df = protein_df[protein_df['residue_number'].isin(resi)]
            pocket_coords = np.array([pocket_df['x_coord'], pocket_df['y_coord'], pocket_df['z_coord']]).T
            pocket_center = np.mean(pocket_coords, axis=0)

        # Swap the 1st and 2nd position of the binding center as the y coord is always before the x
        # coord in the aux_files  and this creates a problem when the pocket_coords and protein_coords
        # are subtracted from the pocket_center
        pocket_center[0], pocket_center[1] = pocket_center[1], pocket_center[0]
        protein_coords = np.array([protein_df['x_coord'], protein_df['y_coord'], protein_df['z_coord']]).T
        pocket_coords = pocket_coords - pocket_center  # center the pocket to 0,0,0
        protein_coords = protein_coords - pocket_center  # center the protein according to the pocket center
        inertia = np.cov(pocket_coords.T)
        e_values, e_vectors = np.linalg.eig(inertia)
        sorted_index = np.argsort(e_values)[::-1]
        sorted_vectors = e_vectors[:, sorted_index]
        # Align the first principal axes to the X-axes
        rx = vrrotvec(np.array([1, 0, 0]), sorted_vectors[:, 0])
        mx = vrrotvec2mat(rx)
        pa1 = np.matmul(mx.T, sorted_vectors)
        # Align the second principal axes to the Y-axes
        ry = vrrotvec(np.array([0, 1, 0]), pa1[:, 1])
        my = vrrotvec2mat(ry)
        transformation_matrix = np.matmul(my.T, mx.T)
        # transform the protein coordinates to the center of the pocket and align with the principal
        # axes with the pocket
        transformed_coords = (np.matmul(transformation_matrix, protein_coords.T)).T
        # Generate a new pdb file with transformed coordinates
        ppdb.df['ATOM']['x_coord'] = transformed_coords[:, 0]
        ppdb.df['ATOM']['y_coord'] = transformed_coords[:, 1]
        ppdb.df['ATOM']['z_coord'] = transformed_coords[:, 2]
        output_trans_pdb_path = aux_input_path[0:-4] + '_trans.pdb'
        print('Output the binding pocket aligned pdb file to: ' + output_trans_pdb_path)
        ppdb.to_pdb(output_trans_pdb_path)

        # Grid generation and DFIRE potential calculation
        print('...Generating pocket grid representation')
        pocket_grid = Grid3DBuilder.build(transformed_coords, output_trans_pdb_path, r, N)
        print('...Generating pocket voxel representation')
        pocket_voxel = site_voxelization(pocket_grid, N + 1)
        pocket_voxel = np.expand_dims(pocket_voxel, axis=0)
        np.save('voxel_rep', pocket_voxel)
        return pocket_voxel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--protein', required=True, help='Protein PDB file path')
    parser.add_argument('-a', '--aux_file', required=True, help='Auxillary file for the protein-ligand complex')
    parser.add_argument('-r', '--radius', type=float, required=True, help='Radius of the spherical grid')
    parser.add_argument('-n', '--number', type=int, required=True,
                        help='The number of points along the dimension of the spherical grid')
    args = parser.parse_args()

    pdb_file_path = args.protein
    aux_file = args.aux_file
    radius = args.radius
    number = args.number

    Vox3DBuilder.voxelization(pdb_file_path, aux_file, radius, number)


if __name__ == "__main__":
    main()

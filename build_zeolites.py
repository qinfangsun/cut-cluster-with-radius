__doc__ = """
A script to take a zeolite framework and guest molecule(s) and carve out a
zeolite cluster based on the location of the guest molecule(s) and distance
cutoff. If Si atoms are not within the distance cutoff but two or more
attached oxygen atoms are, then these Si atoms will also be considered. After
carving out the zeolite framework and guest molecule(s), empty valences will
be filled by a hydrogen atoms (this includes empty valences on the ligand).
Those protons attached to the zeolite framework are then adjusted to have the
desired Si-H or O-H bond lengths (Si-H = 1.4A; O-H = 0.9A).

Copyright Schrodinger LLC, All Rights Reserved.
"""

import argparse
import itertools

from schrodinger.utils import cmdline, fileutils
from schrodinger import structure
from schrodinger.structutils import analyze
from schrodinger.infra import mm, mmbitset
from schrodinger.application.jaguar import input


###############################################################################
def get_merged_st(cmd_args):
    """
    Read the zeolite structure from the input file and merge it with the guest
    structures.

    @param cmd_args:  All script arguments and options.
    @type cmd_args:  class:`argparse.Namespace`

    @return:  Original zeolite structure and the merged structure.
    @rtype:  L{structure.Structure}, L{structure.Structure}
    """

    zeo_st = structure.StructureReader(cmd_args.zeolite_infile).next()

    complex_st = zeo_st.copy()

    for st in structure.StructureReader(cmd_args.guest_infile):
        temp_st = complex_st.merge(st, copy_props=True)
        complex_st = temp_st

    return complex_st, zeo_st


###############################################################################
def generate_atom_lists(complex_st, zeo_st, cmd_args):
    """
    Generate various list of atom indices.

    @param complex_st:  Merged structure of the zeolite and guest molecules.
    @type complex_st:   L{structure.Structure}

    @param zeo_st:  Original zeolite input structure.
    @type zeo_st:   L{structure.Structure}

    @param cmd_args:  All script arguments and options.
    @type cmd_args:  class:`argparse.Namespace`

    @return:  List of close atoms, silicon atoms, oxygen atoms, guest atoms and
    all atoms in the structure
    @rtype:  list, list, list, list, list
    """

    first_guest_atom = zeo_st.atom_total + 1
    last_guest_atom = complex_st.atom_total + 1

    all_atom_list = list(range(1, last_guest_atom))
    guest_atom_list = list(range(first_guest_atom, last_guest_atom))

    atoms_asl = "(atom.num %s)" % ', '.join(map(str, guest_atom_list))
    close_asl = "(within %s %s)" % (str(cmd_args.cutoff), atoms_asl)
    silicon_asl = "atom.element Si"
    oxygen_asl = "atom.element O"

    close_atom_list = analyze.evaluate_asl(complex_st, close_asl)
    silicon_atom_list = analyze.evaluate_asl(complex_st, silicon_asl)
    oxygen_atom_list = analyze.evaluate_asl(complex_st, oxygen_asl)

    return close_atom_list, silicon_atom_list, oxygen_atom_list, \
        guest_atom_list, all_atom_list


###############################################################################
def generate_capping_atoms(atom_list, close_atom_list, complex_st):
    """
    For each atom in the atom list, determine whether it has two or more
    attached atoms that are within the distance cutoff. If this is true and
    the atom is not also in the list of close atoms, add it.

    @param atom_list:  List of all silicon atoms.
    @type atom_list:   list

    @param close_atom_list:  List of close atoms.
    @type close_atom_list:  list

    @param complex_st:  Merged structure of the zeolite and guest molecules.
    @type complex_st:   L{structure.Structure}

    @return:  List of additional capping atom
    @rtype:  list
    """

    extended_atoms = []

    for atom in atom_list:
        if atom in set(close_atom_list):
            continue

        counter = 0
        bonded_atoms = complex_st.atom[atom].bonded_atoms

        for bonded_at in bonded_atoms:
            if bonded_at.index in set(close_atom_list):
                counter += 1

        if counter > 1:
            close_atom_list.append(atom)

    for at in set(close_atom_list):
        extended_atoms.append(at)
        bonded_cap_atoms = complex_st.atom[at].bonded_atoms

        for bonded_cap_at in bonded_cap_atoms:
            extended_atoms.append(bonded_cap_at.index)

    return list(set(extended_atoms))


###############################################################################
def generate_zeolite_cluster(silicon_atom_list,
                             oxygen_atom_list,
                             close_atom_list,
                             all_atom_list,
                             guest_atom_list,
                             complex_st,
                             cmd_args):
    """
    Determine the capping atoms. Find the next shell of atoms around the
    close atoms, store them and delete all other atoms. Convert the last
    shell of atoms to proton and adjust their bond lengths.

    @param silicon_atom_list:  List of all silicon atoms.
    @type silicon_atom_list:   list

    @param oxygen_atom_list:  List of all oxygen atoms.
    @type oxygen_atom_list:   list

    @param close_atom_list:  List of close atoms.
    @type close_atom_list:   list

    @param all_atom_list:  List of all atom in the structure.
    @type all_atom_list:   list

    @param guest_atom_list:  List of all guest atoms in the structure.
    @type guest_atom_list:   list

    @param complex_st:  Merged structure of the zeolite and guest molecules.
    @type complex_st:   L{structure.Structure}

    @param cmd_args:  All script arguments and options.
    @type cmd_args:  class:`argparse.Namespace`

    @return:  List of the final capping atoms based on the new st atom numbers.
    @rtype:  list
    """

    final_cap_at = []

    si_cap_atoms = generate_capping_atoms(silicon_atom_list, close_atom_list,
                                          complex_st)

    ox_cap_atoms = generate_capping_atoms(oxygen_atom_list, close_atom_list,
                                          complex_st)

    extended_atoms = list(set(si_cap_atoms + ox_cap_atoms))

    capping_atoms = list(set(extended_atoms) - set(close_atom_list))

    for subset in itertools.combinations(capping_atoms, 2):
        if complex_st.areBound(subset[0], subset[1]) is True:
            complex_st.deleteBond(subset[0], subset[1])

    delete_atom_list = list(set(all_atom_list) - set(extended_atoms) -
                            set(guest_atom_list))

    atom_map = complex_st.deleteAtoms(delete_atom_list, renumber_map=True)

    for cap_at in capping_atoms:
        if cap_at in atom_map:
            mod_at = complex_st.atom[atom_map[cap_at]]

            if mod_at.element == "O":
                bond_length = 1.4
            else:
                bond_length = 0.9

            final_cap_at.append(mod_at.index)

            mod_at.element = "H"
            mod_at.atom_type = 42
            mod_at.color = "white"

            num_atoms = mm.mmct_ct_get_atom_total(complex_st)
            bs = mmbitset.Bitset(size=num_atoms)

            for bond_at in mod_at.bonded_atoms:
                mm.mmct_atom_set_distance(bond_length, complex_st, bond_at,
                                          complex_st, mod_at, bs)

    writer = structure.StructureWriter(cmd_args.outfile)
    writer.append(complex_st)
    writer.close()

    return complex_st, final_cap_at


###############################################################################
def create_zmat_file(complex_st, final_cap_at, cmd_args):
    """
    For the final structure, create a Jaguar input file with the OH atoms
    constrained.

    @param complex_st:  Merged structure of the zeolite and guest molecules.
    @type complex_st:   L{structure.Structure}

    @param final_cap_at:  List of the final capping atoms.
    @type final_cap_at:  list

    @param cmd_args:  All script arguments and options.
    @type cmd_args:  class:`argparse.Namespace`
    """

    hydrogen_asl = "atom.element 'H'"

    hydrogen_atom_list = analyze.evaluate_asl(complex_st, hydrogen_asl)

    jag_file = fileutils.get_basename(cmd_args.outfile) + ".in"

    jag_obj = input.JaguarInput(structure=complex_st)

    # For each hydrogen, test to see if this is a capping atom. If this is
    # True and that it is bonded to an oxygen, then constrain both atoms.
    # This should ignore OH group in the guest as these atoms should not be
    # in the list of capping atoms.

    for atom in hydrogen_atom_list:
        if atom in set(final_cap_at):
            bonded_atoms = complex_st.atom[atom].bonded_atoms

            for bond_at in bonded_atoms:
                if bond_at.element == 'O':
                    final_cap_at.append(bond_at.index)

    for cap_at in final_cap_at:
        jag_obj.constrainAtomXYZ(complex_st.atom[cap_at])

        complex_st.atom[cap_at].property['b_m_freeze_x'] = 1
        complex_st.atom[cap_at].property['b_m_freeze_y'] = 1
        complex_st.atom[cap_at].property['b_m_freeze_z'] = 1

    jag_obj.saveAs(jag_file)


###############################################################################
def parse_args():
    """
    Parse the command line options.

    @return:  All script arguments and options.
    @rtype:  class:`argparse.Namespace`
    """

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=
                                     argparse.RawDescriptionHelpFormatter)

    parser.add_argument("guest_infile",
                        help="Guest structure file in Maestro format.")

    parser.add_argument("zeolite_infile",
                        help="Zeolite structure file in Maestro format.")

    parser.add_argument("outfile",
                        help="Output file in Maestro format.")

    parser.add_argument("-cutoff",
                        default=3.0,
                        help="Cutoff distance from the guest molecule to "
                             "include from the zeolite structure.")

    parser.add_argument("-jag_out",
                        action='store_true',
                        help="Create a Jaguar output zmat file where OH "
                             "groups are constrained. The basename of the "
                             "output file will be taken from the Maestro "
                             "output filename.")

    args = parser.parse_args()

    return args


###############################################################################
def main():
    """
    Main body of the script.
    """

    cmd_args = parse_args()

    merged_st, zeolite_st = get_merged_st(cmd_args)

    close_atoms, silicon_atoms, oxygen_atoms, guest_atoms, all_atoms = \
        generate_atom_lists(merged_st, zeolite_st, cmd_args)

    final_st, cap_at_list = generate_zeolite_cluster(silicon_atoms,
                                                     oxygen_atoms,
                                                     close_atoms,
                                                     all_atoms,
                                                     guest_atoms,
                                                     merged_st,
                                                     cmd_args)

    if cmd_args.jag_out:
        create_zmat_file(final_st, cap_at_list, cmd_args)

if __name__ == '__main__':
    cmdline.main_wrapper(main)

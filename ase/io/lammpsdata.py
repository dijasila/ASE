import re
import numpy as np
from ase.atoms import Atoms
from ase.quaternions import Quaternions
from ase.parallel import paropen
from ase.utils import basestring
from ase.calculators.lammps import Prism, convert


def read_lammps_data(fileobj, Z_of_type=None, style="full",
                     sort_by_id=False, units="metal"):
    """Method which reads a LAMMPS data file.

    sort_by_id: Order the particles according to their id. Might be faster to
    switch it off.
    Units are set by default to the style=metal setting in LAMMPS.
    """
    if isinstance(fileobj, basestring):
        f = paropen(fileobj)
    else:
        f = fileobj

    # load everything into memory
    lines = f.readlines()

    # begin read_lammps_data
    comment = None
    N = None
    # N_types = None
    xlo = None
    xhi = None
    ylo = None
    yhi = None
    zlo = None
    zhi = None
    xy = None
    xz = None
    yz = None
    pos_in = {}
    travel_in = {}
    mol_id_in = {}
    charge_in = {}
    mass_in = {}
    name_in = {}
    resname_in = {}
    vel_in = {}
    bonds_in = []
    angles_in = []
    dihedrals_in = []
    impropers_in = []

    sections = [
        "Atoms",
        "Velocities",
        "Masses",
        "Charges",
        "Ellipsoids",
        "Lines",
        "Triangles",
        "Bodies",
        "Bonds",
        "Angles",
        "Dihedrals",
        "Impropers",
        "Impropers Pair Coeffs",
        "PairIJ Coeffs",
        "Pair Coeffs",
        "Bond Coeffs",
        "Angle Coeffs",
        "Dihedral Coeffs",
        "Improper Coeffs",
        "BondBond Coeffs",
        "BondAngle Coeffs",
        "MiddleBondTorsion Coeffs",
        "EndBondTorsion Coeffs",
        "AngleTorsion Coeffs",
        "AngleAngleTorsion Coeffs",
        "BondBond13 Coeffs",
        "AngleAngle Coeffs",
    ]
    header_fields = [
        "atoms",
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "atom types",
        "bond types",
        "angle types",
        "dihedral types",
        "improper types",
        "extra bond per atom",
        "extra angle per atom",
        "extra dihedral per atom",
        "extra improper per atom",
        "extra special per atom",
        "ellipsoids",
        "lines",
        "triangles",
        "bodies",
        "xlo xhi",
        "ylo yhi",
        "zlo zhi",
        "xy xz yz",
    ]
    sections_re = "(" + "|".join(sections).replace(" ", "\\s+") + ")"
    header_fields_re = "(" + "|".join(header_fields).replace(" ", "\\s+") + ")"

    section = None
    header = True
    for line in lines:
        if comment is None:
            comment = line.rstrip()
        else:
            line_comment = None
            if '#' in line:
                line_comment = line.split('#')[1]
            line = re.sub("#.*", "", line).rstrip().lstrip()
            line = line.rstrip().lstrip()
            if re.match("^\\s*$", line):  # skip blank lines
                continue

        # check for known section names
        m = re.match(sections_re, line)
        if m is not None:
            section = m.group(0).rstrip().lstrip()
            header = False
            continue

        if header:
            field = None
            val = None
            # m = re.match(header_fields_re+"\s+=\s*(.*)", line)
            # if m is not None: # got a header line
            #   field=m.group(1).lstrip().rstrip()
            #   val=m.group(2).lstrip().rstrip()
            # else: # try other format
            #   m = re.match("(.*)\s+"+header_fields_re, line)
            #   if m is not None:
            #       field = m.group(2).lstrip().rstrip()
            #       val = m.group(1).lstrip().rstrip()
            m = re.match("(.*)\\s+" + header_fields_re, line)
            if m is not None:
                field = m.group(2).lstrip().rstrip()
                val = m.group(1).lstrip().rstrip()
            if field is not None and val is not None:
                if field == "atoms":
                    N = int(val)
                # elif field == "atom types":
                #     N_types = int(val)
                elif field == "xlo xhi":
                    (xlo, xhi) = [float(x) for x in val.split()]
                elif field == "ylo yhi":
                    (ylo, yhi) = [float(x) for x in val.split()]
                elif field == "zlo zhi":
                    (zlo, zhi) = [float(x) for x in val.split()]
                elif field == "xy xz yz":
                    (xy, xz, yz) = [float(x) for x in val.split()]

        if section is not None:
            fields = line.split()
            if section == "Atoms":  # id *
                id = int(fields[0])
                if style == "full" and (len(fields) == 7 or len(fields) == 10):
                    # id mol-id type q x y z [tx ty tz]
                    pos_in[id] = (
                        int(fields[2]),
                        float(fields[4]),
                        float(fields[5]),
                        float(fields[6]),
                    )
                    mol_id_in[id] = int(fields[1])
                    charge_in[id] = float(fields[3])
                    if len(fields) == 10:
                        travel_in[id] = (
                            int(fields[7]),
                            int(fields[8]),
                            int(fields[9]),
                        )
                elif style == "atomic" and (
                        len(fields) == 5 or len(fields) == 8
                ):
                    # id type x y z [tx ty tz]
                    pos_in[id] = (
                        int(fields[1]),
                        float(fields[2]),
                        float(fields[3]),
                        float(fields[4]),
                    )
                    if len(fields) == 8:
                        travel_in[id] = (
                            int(fields[5]),
                            int(fields[6]),
                            int(fields[7]),
                        )
                elif (style in ("angle", "bond", "molecular")
                      ) and (len(fields) == 6 or len(fields) == 9):
                    # id mol-id type x y z [tx ty tz]
                    pos_in[id] = (
                        int(fields[2]),
                        float(fields[3]),
                        float(fields[4]),
                        float(fields[5]),
                    )
                    mol_id_in[id] = int(fields[1])
                    if len(fields) == 9:
                        travel_in[id] = (
                            int(fields[6]),
                            int(fields[7]),
                            int(fields[8]),
                        )
                else:
                    raise RuntimeError("Style '{}' not supported or invalid "
                                       "number of fields {}"
                                       "".format(style, len(fields)))
                if line_comment:
                    resname_in[id] = set(line_comment.split())
            elif section == "Velocities":  # id vx vy vz
                vel_in[int(fields[0])] = (
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                )
            elif section == "Masses":
                mass_in[int(fields[0])] = float(fields[1])
                if line_comment:
                    name_in[int(fields[0])] = line_comment.split()[0]
            elif section == "Bonds":  # id type atom1 atom2
                bonds_in.append(
                    (int(fields[1]), int(fields[2]), int(fields[3]))
                )
            elif section == "Angles":  # id type atom1 atom2 atom3
                angles_in.append(
                    (
                        int(fields[1]),
                        int(fields[2]),
                        int(fields[3]),
                        int(fields[4]),
                    )
                )
            elif section == "Dihedrals":  # id type atom1 atom2 atom3 atom4
                dihedrals_in.append(
                    (
                        int(fields[1]),
                        int(fields[2]),
                        int(fields[3]),
                        int(fields[4]),
                        int(fields[5]),
                    )
                )
            elif section == "Impropers":  # id type atom1 atom2 atom3 atom4
                impropers_in.append(
                    (
                        int(fields[1]),
                        int(fields[2]),
                        int(fields[3]),
                        int(fields[4]),
                        int(fields[5]),
                    )
                )

    # set cell
    cell = np.zeros((3, 3))
    cell[0, 0] = xhi - xlo
    cell[1, 1] = yhi - ylo
    cell[2, 2] = zhi - zlo
    if xy is not None:
        cell[1, 0] = xy
    if xz is not None:
        cell[2, 0] = xz
    if yz is not None:
        cell[2, 1] = yz

    # initialize arrays for per-atom quantities
    positions = np.zeros((N, 3))
    numbers = np.zeros((N), int)
    ids = np.zeros((N), int)
    types = np.zeros((N), int)
    if len(vel_in) > 0:
        velocities = np.zeros((N, 3))
    else:
        velocities = None
    if len(mass_in) > 0:
        masses = np.zeros((N))
    else:
        masses = None
    if len(mol_id_in) > 0:
        mol_id = np.zeros((N), int)
    else:
        mol_id = None
    if len(charge_in) > 0:
        charge = np.zeros((N), float)
    else:
        charge = None
    if len(travel_in) > 0:
        travel = np.zeros((N, 3), int)
    else:
        travel = None
    if len(name_in) > 0:
        names = np.empty((N), object)
    else:
        names = None
    if len(resname_in) > 0:
        resnames = np.empty((N), object)
    else:
        resnames = None
    if len(bonds_in) > 0:
        bonds = [{} for _ in range(N)]
    else:
        bonds = None
    if len(angles_in) > 0:
        angles = [{} for _ in range(N)]
    else:
        angles = None
    if len(dihedrals_in) > 0:
        dihedrals = [{} for _ in range(N)]
    else:
        dihedrals = None
    if len(impropers_in) > 0:
        impropers = [{} for _ in range(N)]
    else:
        impropers = None

    ind_of_id = {}
    # copy per-atom quantities from read-in values
    for (i, id) in enumerate(pos_in.keys()):
        # by id
        ind_of_id[id] = i
        if sort_by_id:
            ind = id - 1
        else:
            ind = i
        type = pos_in[id][0]
        positions[ind, :] = [pos_in[id][1], pos_in[id][2], pos_in[id][3]]
        if velocities is not None:
            velocities[ind, :] = [vel_in[id][0], vel_in[id][1], vel_in[id][2]]
        if travel is not None:
            travel[ind] = travel_in[id]
        if mol_id is not None:
            mol_id[ind] = mol_id_in[id]
        if resnames is not None:
            resnames[ind] = resname_in[id]
        if charge is not None:
            charge[ind] = charge_in[id]
        ids[ind] = id
        # by type
        types[ind] = type
        if Z_of_type is None:
            numbers[ind] = type
        else:
            numbers[ind] = Z_of_type[type]
        if names is not None:
            names[ind] = name_in[type]
        if masses is not None:
            masses[ind] = mass_in[type]
    # convert units
    positions = convert(positions, "distance", units, "ASE")
    cell = convert(cell, "distance", units, "ASE")
    if masses is not None:
        masses = convert(masses, "mass", units, "ASE")
    if velocities is not None:
        velocities = convert(velocities, "velocity", units, "ASE")

    # create ase.Atoms
    at = Atoms(positions=positions,
               numbers=numbers,
               masses=masses,
               cell=cell,
               pbc=[True, True, True])
    # set velocities (can't do it via constructor)
    if velocities is not None:
        at.set_velocities(velocities)
    at.arrays["ids"] = ids
    at.arrays["types"] = types
    if names is not None:
        at.arrays["names"] = names
    if travel is not None:
        at.arrays["travel"] = travel
    if mol_id is not None:
        at.arrays["mol-ids"] = mol_id
    if charge is not None:
        at.arrays["initial_charges"] = charge
        at.arrays["mmcharges"] = charge.copy()
    if resnames is not None:
        # removing names from resnames
        # if names exist
        if names is not None:
            resnames -= np.asarray([set([i]) for i in names])
        at.set_array('resnames', resnames, object)

    if bonds is not None:
        for (type, a1, a2) in bonds_in:
            i_a1 = ind_of_id[a1]
            i_a2 = ind_of_id[a2]
            # Double list for bonds to make it consistent with other sections
            bonds[i_a1][type] = bonds[i_a1].get(type, []) + [[i_a2]]
        at.set_array('bonds', bonds, 'object')

    if angles is not None:
        for (type, a1, a2, a3) in angles_in:
            i_a1 = ind_of_id[a1]
            i_a2 = ind_of_id[a2]
            i_a3 = ind_of_id[a3]
            angles[i_a2][type] = angles[i_a2].get(type, []) + [[i_a1, i_a3]]
        at.set_array('angles', angles, 'object')

    if dihedrals is not None:
        for (type, a1, a2, a3, a4) in dihedrals_in:
            i_a1 = ind_of_id[a1]
            i_a2 = ind_of_id[a2]
            i_a3 = ind_of_id[a3]
            i_a4 = ind_of_id[a4]
            dihedrals[i_a1][type] = dihedrals[i_a1].get(type, []) \
                + [[i_a2, i_a3, i_a4]]
        at.set_array('dihedrals', dihedrals, 'object')

    if impropers is not None:
        for (type, a1, a2, a3, a4) in impropers_in:
            i_a1 = ind_of_id[a1]
            i_a2 = ind_of_id[a2]
            i_a3 = ind_of_id[a3]
            i_a4 = ind_of_id[a4]
            impropers[i_a1][type] = impropers[i_a1].get(type, []) \
                + [[i_a2, i_a3, i_a4]]
        at.set_array('impropers', impropers, 'object')

    at.info["comment"] = comment
    at.topology.update()

    return at


def write_lammps_data(fileobj, atoms, specorder=None, force_skew=False,
                      prismobj=None, velocities=False, units="metal",
                      style='atomic'):
    """Write atomic structure data to a LAMMPS data_ file."""
    if isinstance(fileobj, basestring):
        f = paropen(fileobj, "wb")
        close_file = True
    else:
        # Presume fileobj acts like a fileobj
        f = fileobj
        close_file = False

    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError(
                'Can only write one configuration to a lammps data file!')
        atoms = atoms[0]

    # TODO: add quarternions printing
    if not atoms.has('ids'):
        atoms.set_topology()

    if specorder is not None:
        # To index elements in the LAMMPS data file
        # (indices must correspond to order in the potential file)
        atoms.topology.update(specorder=specorder)

    f.write('{0} (written by ASE) \n\n'.format(f.name).encode("utf-8"))

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    f.write('{0:8} \t atoms \n'.format(n_atoms).encode("utf-8"))
    for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
        try:
            num = atoms.topology[prop].get_count()
        except KeyError:
            num = 0
        f.write('{0:8} \t '
                '{1} \n'.format(num,
                                prop).encode("utf-8"))

    # TopoAtoms always assigns type and name
    types = atoms.get_array('types')
    names = atoms.get_array('names')

    n_types = len(np.unique(atoms.get_array('types')))
    f.write('{0:8} \t '
            'atom types\n'.format(n_types).encode("utf-8"))
    for prop in ['bond types', 'angle types', 'dihedral types',
                 'improper types']:
        try:
            num = atoms.topology[prop.split()[0] + 's'].get_num_types()
        except KeyError:
            num = 0
        f.write('{0:8} \t '
                '{1} \n'.format(num,
                                prop).encode("utf-8"))
    if prismobj is None:
        p = Prism(atoms.get_cell())
    else:
        p = prismobj

    # Get cell parameters and convert from ASE units to LAMMPS units
    xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(), "distance",
                                        "ASE", units)

    f.write("\t 0.0 {0:12.6f}  xlo xhi\n".format(xhi).encode("utf-8"))
    f.write("\t 0.0 {0:12.6f}  ylo yhi\n".format(yhi).encode("utf-8"))
    f.write("\t 0.0 {0:12.6f}  zlo zhi\n".format(zhi).encode("utf-8"))

    if force_skew or p.is_skewed():
        f.write(
            "{0:12.6f} {1:12.6f} {2:12.6f}  xy xz yz\n".format(
                xy, xz, yz
            ).encode("utf-8")
        )
    f.write("\n\n".encode("utf-8"))

    f.write('Masses \n\n'.encode("utf-8"))
    for i in np.unique(atoms.get_array('types')):
        indx = np.where(i == types)[0][0]
        mass = atoms.get_masses()[indx]
        sym = names[indx]
        f.write('{0:>6} {1:8.4f} # {2}\n'.format(
                i, mass, sym
            ).encode("utf-8")
        )
    f.write('\n\n'.encode("utf-8"))

    f.write('Atoms \n\n'.encode("utf-8"))
    if atoms.has('travel'):
        travel = atoms.get_array('travel')
    else:
        travel = None
    if atoms.has('resnames'):
        resnames = atoms.get_array('resnames')
    else:
        resnames = None
    id = atoms.get_array('ids')
    mol_id = atoms.get_array('mol-ids')
    pos = p.vector_to_lammps(atoms.get_positions(), wrap=True)
    charges = atoms.get_initial_charges()
    if style == 'full':
        # id mol-id type q x y z [tx ty tz]
        for i, (q, r) in enumerate(zip(charges, pos)):
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            f.write('{0:6} {1:4} {2:4} {3:8.6f}'
                    ' {4:12.6f} {5:12.6f} {6:12.6f}'.format(
                                          id[i],
                                          mol_id[i],
                                          types[i],
                                          q,
                                          *r
                        ).encode("utf-8")
                    )
            if travel:
                f.write(' {0:8.4f} {1:8.4f} '
                        '{2:8.4f}'.format(*travel[i]).encode("utf-8")
                        )
            f.write(' # {0}'.format(names[i]).encode("utf-8")
                    )
            if resnames is not None:
                f.write(' {0}'.format(' '.join(list(resnames[i]))
                                      ).encode("utf-8"))
            f.write('\n'.encode("utf-8"))
    elif style == 'charge':
        # id type q x y z [tx ty tz]
        for i, (q, r) in enumerate(zip(charges, pos)):
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            f.write('{0:6} {1:4} {2:8.6f}'
                    ' {3:12.6f} {4:12.6f} {5:12.6f}'.format(
                                          id[i],
                                          mol_id[i],
                                          types[i],
                                          q,
                                          *r
                        ).encode("utf-8")
                    )
            if travel:
                f.write(' {0:8.4f} {1:8.4f} '
                        '{2:8.4f}'.format(*travel[i]).encode("utf-8")
                        )
            f.write(' # {0}'.format(names[i]).encode("utf-8")
                    )
            if resnames is not None:
                f.write(' {0}'.format(' '.join(list(resnames[i]))
                                      ).encode("utf-8"))
            f.write('\n'.encode("utf-8"))
    elif style == 'atomic':
        # id type x y z [tx ty tz]
        for i, r in enumerate(pos):
            f.write('{0:6} {1:4}'
                    ' {2:12.6f} {3:12.6f} {4:12.6f}'.format(
                                          id[i],
                                          types[i],
                                          *r
                        ).encode("utf-8")
                    )
            if travel:
                f.write(' {0:8.4f} {1:8.4f} '
                        '{2:8.4f}'.format(*travel[i]).encode("utf-8")
                        )
            f.write(' # {0}'.format(names[i]).encode("utf-8")
                    )
            if resnames is not None:
                f.write(' {0}'.format(' '.join(list(resnames[i]))
                                      ).encode("utf-8"))
            f.write('\n'.encode("utf-8"))
    elif (style == 'angle' or style == 'bond' or
          style == 'molecular'):
        # id mol-id type x y z [tx ty tz]
        for i, r in enumerate(pos):
            f.write('{0:6} {1:4} {2:4}'
                    ' {3:12.6f} {4:12.6f} {5:12.6f}'.format(
                                          id[i],
                                          mol_id[i],
                                          types[i],
                                          *r
                        ).encode("utf-8")
                    )
            if travel:
                f.write(' {0:8.4f} {1:8.4f} '
                        '{2:8.4f}'.format(*travel[i]).encode("utf-8")
                        )
            f.write(' # {0}'.format(names[i]).encode("utf-8")
                    )
            if resnames is not None:
                f.write(' {0}'.format(' '.join(list(resnames[i]))
                                      ).encode("utf-8"))
            f.write('\n'.encode("utf-8"))
    else:
        raise NotImplementedError('style {0} not supported. '
                                  'Use: full, atomic, angle, bond, charge,'
                                  ' or molecular'.format(style))
    f.write('\n\n'.encode("utf-8"))

    if atoms.topology['bonds'].get_count():
        typ = atoms.topology['bonds'].get_types()
        count = 1
        f.write('Bonds \n\n'.encode("utf-8"))
        for indx, item in enumerate(atoms.get_array('bonds')):
            for key, values in item.items():
                for value in values:
                    f.write('{0:6} {1:6} {2:6} '
                            '{3:6}\t# {4}\n'.format(count,
                                                    key,
                                                    indx + 1,
                                                    value[0] + 1,
                                                    typ[key]
                                                    ).encode("utf-8")
                            )
                    count += 1
        f.write('\n\n'.encode("utf-8"))

    if atoms.topology['angles'].get_count():
        typ = atoms.topology['angles'].get_types()
        count = 1
        f.write('Angles \n\n'.encode("utf-8"))
        for indx, item in enumerate(atoms.get_array('angles')):
            for key, values in item.items():
                for value in values:
                    f.write('{0:6} {1:6} {2:6}'
                            '{3:6} {4:6}'
                            '\t# {5}\n'.format(count,
                                               key,
                                               value[0] + 1,
                                               indx + 1,
                                               value[1] + 1,
                                               typ[key]
                                               ).encode("utf-8")
                            )
                    count += 1
        f.write('\n\n'.encode("utf-8"))

    if atoms.topology['dihedrals'].get_count():
        typ = atoms.topology['dihedrals'].get_types()
        count = 1
        f.write('Dihedrals \n\n'.encode("utf-8"))
        for indx, item in enumerate(atoms.get_array('dihedrals')):
            for key, values in item.items():
                for value in values:
                    _ = [i + 1 for i in value]
                    f.write('{0:6} {1:6} {2:6} {3:6} {4:6} '
                            '{5:6}'.format(count,
                                           key,
                                           indx + 1,
                                           *_).encode("utf-8")
                            )
                    f.write('\t# {0}\n'.format(typ[key]
                                               ).encode("utf-8")
                            )
                    count += 1
        f.write('\n\n'.encode("utf-8"))

    if atoms.topology['impropers'].get_count():
        typ = atoms.topology['impropers'].get_types()
        count = 1
        f.write('Impropers \n\n'.encode("utf-8"))
        for indx, item in enumerate(atoms.get_array('impropers')):
            for key, values in item.items():
                for value in values:
                    _ = [i + 1 for i in value]
                    f.write('{0:6} {1:6} {2:6} {3:6} {4:6} '
                            '{5:6}'.format(count,
                                           key,
                                           indx + 1,
                                           *_
                                           ).encode("utf-8")
                            )
                    f.write('\t# {0}\n'.format(typ[key]
                                               ).encode("utf-8")
                            )
                    count += 1
        f.write('\n\n'.encode("utf-8"))

    if velocities and atoms.get_velocities() is not None:
        f.write("\n\nVelocities \n\n".encode("utf-8"))
        vel = p.vector_to_lammps(atoms.get_velocities())
        for i, v in enumerate(vel):
            # Convert velocity from ASE units to LAMMPS units
            v = convert(v, "velocity", "ASE", units)
            f.write('{0:>6} {1:12.6f} {2:12.6f} {3:12.6f}\n'.format(
                        *(i + 1,) + tuple(v)).encode("utf-8")
                    )

    f.flush()
    if close_file:
        f.close()

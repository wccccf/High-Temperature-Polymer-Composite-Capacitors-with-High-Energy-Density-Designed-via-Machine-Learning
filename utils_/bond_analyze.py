# Bond lengths from:
# http://www.wiredchemist.com/chemistry/data/bond_energies_lengths.html
# And:
# http://chemistry-reference.com/tables/Bond%20Lengths%20and%20Enthalpies.pdf
bonds1 = {'H': {'H': 74, 'C': 109, 'N': 101, 'O': 96, 'F': 92,
                'Si': 148, 'P': 144,  'S': 134,
                'Cl': 127, 'Br': 141},
          'C': {'H': 109, 'C': 154, 'N': 147, 'O': 143, 'F': 135,
                'Si': 185, 'P': 184, 'S': 182, 'Cl': 177, 'Br': 194},
          'N': {'H': 101, 'C': 147, 'N': 145, 'O': 140, 'F': 136,
                'Cl': 175, 'Br': 214, 'S': 168,  'P': 177,
                'Si': 180},
          'O': {'H': 96, 'C': 143, 'N': 140, 'O': 148, 'F': 142,
                'Br': 172, 'S': 151, 'P': 163, 'Si': 163, 'Cl': 164},
          'F': {'H': 92, 'C': 135, 'N': 136, 'O': 142, 'F': 142,
                'S': 158, 'Si': 160, 'Cl': 166, 'Br': 178, 'P': 156},
          'Si': {'Si': 233, 'H': 148, 'C': 185, 'O': 163, 'S': 200,
                 'F': 160, 'Cl': 202, 'Br': 215, 'N': 180,
                 'P': 245},
          'Cl': {'Cl': 199, 'H': 127, 'C': 177, 'N': 175, 'O': 164,
                 'P': 203, 'S': 207, 'Si': 202, 'F': 166,
                 'Br': 214},
          'S': {'H': 134, 'C': 182, 'N': 168, 'O': 151, 'S': 204,
                'F': 158, 'Cl': 207, 'Br': 225, 'Si': 200, 'P': 210},
          'Br': {'Br': 228, 'H': 141, 'C': 194, 'O': 172, 'N': 214,
                 'Si': 215, 'S': 225, 'F': 178, 'Cl': 214, 'P': 222},
          'P': {'P': 221, 'H': 144, 'C': 184, 'O': 163, 'Cl': 203,
                'S': 210, 'F': 156, 'N': 177, 'Br': 222,
                'Si': 245},
          }

bonds2 = {'C': {'C': 134, 'N': 129, 'O': 120, 'S': 160, 'Si': 171, 'P': 168},
          'N': {'C': 129, 'N': 125, 'O': 121, 'S': 167, 'Si': 162, 'P': 160},
          'O': {'C': 120, 'N': 121, 'O': 121, 'P': 150, 'S': 163},
          'P': {'O': 150, 'S': 186, 'C': 168, 'N': 160},
          'S': {'P': 186, 'O': 163, 'N': 167, 'C': 160},
          'Si': {'Si': 231, 'C': 171, 'N': 162}}

bonds3 = {'C': {'C': 120, 'N': 116, 'O': 113, 'P': 158},
          'N': {'C': 116, 'N': 110},
          'O': {'C': 113},
          'P': {'C': 158}}

margin1, margin2, margin3 = 10, 5, 3

allowed_bonds = {'H': 1, 'C': 4, 'N': 3, 'O': 2, 'F': 1,
                 'Si': 4, 'P': [3, 5],
                 'S': [2, 4 , 6], 'Cl': 1, 'Br': 1, 'I': 1, }


def get_bond_order(atom1, atom2, distance, check_exists=False):
    distance = 100 * distance

    # Check exists for large molecules where some atom pairs do not have a
    if check_exists:
        if atom1 not in bonds1:
            return 0
        if atom2 not in bonds1[atom1]:
            return 0

    if distance < bonds1[atom1][atom2] + margin1:

        # Check if atoms in bonds2 dictionary.
        if atom1 in bonds2 and atom2 in bonds2[atom1]:
            thr_bond2 = bonds2[atom1][atom2] + margin2
            if distance < thr_bond2:
                if atom1 in bonds3 and atom2 in bonds3[atom1]:
                    thr_bond3 = bonds3[atom1][atom2] + margin3
                    if distance < thr_bond3:
                        return 3        # Triple
                return 2            # Double
        return 1                # Single
    return 0                    # No bond

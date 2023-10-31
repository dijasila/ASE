# flake8: noqa


# this returns a look up dict for the orbital name -> col number in the
# DOSCAR file based on the number of columns found
PDOS_orbital_names_and_DOSCAR_column = {
    4: {'s': 1, 'p': 2, 'd': 3},
    5: {'s': 1, 'p': 2, 'd': 3, 'f': 4},
    7: {
        's+': 1,
        's-up': 1,
        's-': 2,
        's-down': 2,
        'p+': 3,
        'p-up': 3,
        'p-': 4,
        'p-down': 4,
        'd+': 5,
        'd-up': 5,
        'd-': 6,
        'd-down': 6,
    },
    9: {
        's+': 1,
        's-up': 1,
        's-': 2,
        's-down': 2,
        'p+': 3,
        'p-up': 3,
        'p-': 4,
        'p-down': 4,
        'd+': 5,
        'd-up': 5,
        'd-': 6,
        'd-down': 6,
        'f+': 7,
        'f-up': 7,
        'f-': 8,
        'f-down': 8,
    },
    10: {
        's': 1,
        'py': 2,
        'pz': 3,
        'px': 4,
        'dxy': 5,
        'dyz': 6,
        'dz2': 7,
        'dxz': 8,
        'dx2': 9,
    },
    19: {
        's+': 1,
        's-up': 1,
        's-': 2,
        's-down': 2,
        'py+': 3,
        'py-up': 3,
        'py-': 4,
        'py-down': 4,
        'pz+': 5,
        'pz-up': 5,
        'pz-': 6,
        'pz-down': 6,
        'px+': 7,
        'px-up': 7,
        'px-': 8,
        'px-down': 8,
        'dxy+': 9,
        'dxy-up': 9,
        'dxy-': 10,
        'dxy-down': 10,
        'dyz+': 11,
        'dyz-up': 11,
        'dyz-': 12,
        'dyz-down': 12,
        'dz2+': 13,
        'dz2-up': 13,
        'dz2-': 14,
        'dz2-down': 14,
        'dxz+': 15,
        'dxz-up': 15,
        'dxz-': 16,
        'dxz-down': 16,
        'dx2+': 17,
        'dx2-up': 17,
        'dx2-': 18,
        'dx2-down': 18,
    },
    17: {
        's': 1,
        'py': 2,
        'pz': 3,
        'px': 4,
        'dxy': 5,
        'dyz': 6,
        'dz2': 7,
        'dxz': 8,
        'dx2': 9,
        'fy(3x2-y2)': 10,
        'fxyz': 11,
        'fyz2': 12,
        'fz3': 13,
        'fxz2': 14,
        'fz(x2-y2)': 15,
        'fx(x2-3y2)': 16,
    },
    19: {
        's+': 1,
        's-up': 1,
        's-': 2,
        's-down': 2,
        'py+': 3,
        'py-up': 3,
        'py-': 4,
        'py-down': 4,
        'pz+': 5,
        'pz-up': 5,
        'pz-': 6,
        'pz-down': 6,
        'px+': 7,
        'px-up': 7,
        'px-': 8,
        'px-down': 8,
        'dxy+': 9,
        'dxy-up': 9,
        'dxy-': 10,
        'dxy-down': 10,
        'dyz+': 11,
        'dyz-up': 11,
        'dyz-': 12,
        'dyz-down': 12,
        'dz2+': 13,
        'dz2-up': 13,
        'dz2-': 14,
        'dz2-down': 14,
        'dxz+': 15,
        'dxz-up': 15,
        'dxz-': 16,
        'dxz-down': 16,
        'dx2+': 17,
        'dx2-up': 17,
        'dx2-': 18,
        'dx2-down': 18,
    },
    # this is Non-collinear. -x, -y, -z are magnetic moment
    # vasp reports totals for each orbital first
    37: {
        's': 1,
        's-x': 2,
        's-y': 3,
        's-z': 4,
        'py': 5,
        'py-x': 6,
        'py-y': 7,
        'py-z': 8,
        'pz': 9,
        'pz-x': 10,
        'pz-y': 11,
        'pz-z': 12,
        'px': 13,
        'px-x': 14,
        'px-y': 15,
        'px-z': 16,
        'dxy': 17,
        'dxy-x': 18,
        'dxy-y': 19,
        'dxy-z': 20,
        'dyz': 21,
        'dyz-x': 22,
        'dyz-y': 23,
        'dyz-z': 24,
        'dz2': 25,
        'dz2-x': 26,
        'dz2-y': 27,
        'dz2-z': 28,
        'dxz': 29,
        'dxz-x': 30,
        'dxz-y': 31,
        'dxz-z': 32,
        'dx2': 33,
        'dx2-x': 34,
        'dx2-y': 35,
        'dx2-z': 36,
    },
    33: {
        's+': 1,
        's-up': 1,
        's-': 2,
        's-down': 2,
        'py+': 3,
        'py-up': 3,
        'py-': 4,
        'py-down': 4,
        'pz+': 5,
        'pz-up': 5,
        'pz-': 6,
        'pz-down': 6,
        'px+': 7,
        'px-up': 7,
        'px-': 8,
        'px-down': 8,
        'dxy+': 9,
        'dxy-up': 9,
        'dxy-': 10,
        'dxy-down': 10,
        'dyz+': 11,
        'dyz-up': 11,
        'dyz-': 12,
        'dyz-down': 12,
        'dz2+': 13,
        'dz2-up': 13,
        'dz2-': 14,
        'dz2-down': 14,
        'dxz+': 15,
        'dxz-up': 15,
        'dxz-': 16,
        'dxz-down': 16,
        'dx2+': 17,
        'dx2-up': 17,
        'dx2-': 18,
        'dx2-down': 18,
        'fy(3x2-y2)+': 19,
        'fy(3x2-y2)-up': 19,
        'fy(3x2-y2)-': 20,
        'fy(3x2-y2)-down': 20,
        'fxyz+': 21,
        'fxyz-up': 21,
        'fxyz-': 22,
        'fxyz-down': 22,
        'fyz2+': 23,
        'fyz2-up': 23,
        'fyz2-': 24,
        'fyz2-down': 24,
        'fz3+': 25,
        'fz3-up': 25,
        'fz3-': 26,
        'fz3-down': 26,
        'fxz2+': 27,
        'fxz2-up': 27,
        'fxz2-': 28,
        'fxz2-down': 28,
        'fz(x2-y2)+': 29,
        'fz(x2-y2)-up': 29,
        'fz(x2-y2)-': 30,
        'fz(x2-y2)-down': 30,
        'fx(x2-3y2)+': 31,
        'fx(x2-3y2)-up': 31,
        'fx(x2-3y2)-': 32,
        'fx(x2-3y2)-down': 32,
    },
}

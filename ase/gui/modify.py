from functools import partial

from ase.gui.i18n import _

import ase.gui.ui as ui
from ase.gui.widgets import Element
from ase.gui.utils import get_magmoms


class ModifyAtoms:
    """Presents a dialog box where the user is able to change the
    atomic type, the magnetic moment and tags of the selected atoms,
    and switch atomic indices in preparation of input for NEB calculations.
    """
    def __init__(self, gui):
        self.gui = gui
        selected = self.selection()
        if not selected.any():
            ui.error(_('No atoms selected!'))
            return

        win = ui.Window(_('Modify'))
        element = Element(callback=self.set_element)
        win.add(element)
        win.add(ui.Button(_('Change element'),
                          partial(self.set_element, element)))
        self.tag = ui.SpinBox(0, -1000, 1000, 1, self.set_tag)
        win.add([_('Tag'), self.tag])
        self.magmom = ui.SpinBox(0.0, -10, 10, 0.1, self.set_magmom)
        win.add([_('Moment'), self.magmom])

        atoms = self.gui.atoms
        sym = atoms.symbols[selected]
        if len(sym.species()) == 1:
            element.symbol = sym[0]

        tags = atoms.get_tags()[selected]
        if tags.ptp() == 0:
            self.tag.value = tags[0]

        magmoms = get_magmoms(atoms)[selected]
        if magmoms.round(2).ptp() == 0.0:
            self.magmom.value = round(magmoms[0], 2)

        win.add(ui.Button(_('Switch indices'), self.switch_indices))
        win.add(ui.Button(_('Sort indices by element'), self.sort_indices_by_element))

    def switch_indices(self):
        import numpy as np
        selected = self.selection()
        indices = np.where(selected == True)[0]
        selection_len = len([i for i in selected if i == True])
        if not selection_len == 2:
            ui.error(_('Only two atoms must be selected!'))
            return

        # Defining list of manipulated atoms
        i = [atom.index for atom in self.gui.atoms]
        i[indices[0]], i[indices[1]] = i[indices[1]], i[indices[0]]
        self.gui.new_atoms(self.gui.atoms[i])

    def sort_indices_by_element(self):
        import numpy as np
        ordered = []
        symbols = self.gui.atoms.get_chemical_symbols()
        symbols_set = sorted(list(set(symbols)))
        for i in symbols_set:
            ordered += [x.index for x in self.gui.atoms if x.symbol == i]
        self.gui.new_atoms(self.gui.atoms[ordered])

    def selection(self):
        return self.gui.images.selected[:len(self.gui.atoms)]

    def set_element(self, element):
        self.gui.atoms.numbers[self.selection()] = element.Z
        self.gui.draw()

    def set_tag(self):
        tags = self.gui.atoms.get_tags()
        tags[self.selection()] = self.tag.value
        self.gui.atoms.set_tags(tags)
        self.gui.draw()

    def set_magmom(self):
        magmoms = get_magmoms(self.gui.atoms)
        magmoms[self.selection()] = self.magmom.value
        self.gui.atoms.set_initial_magnetic_moments(magmoms)
        self.gui.draw()

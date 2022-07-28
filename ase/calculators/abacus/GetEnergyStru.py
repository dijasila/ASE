import os
import re
from ase import Atoms
from ase.io import write
   
# Set your working path which the position of you running code
WorkPath = os.getcwd() + '/'


def OpenFile(FileName = 'STRU_NOW.cif'):
    # Open file
    try:
        FileOne = open(FileName, 'r') 
        Lines = FileOne.readlines()
        FileOne.close()
        return Lines
    except:
        # print('Open file error')
        return False


def GetStruOne():
    # Get atoms object from abacus 'STRU_NOW.cif' file  
    Lines = OpenFile('STRU_NOW.cif')
    StruCell = list()
    StruSymbol = list()
    StruScalePos = list()
    FindPos = False
    for Line in Lines:
        # Locate position
        if(len(Line.split())== 1): 
            if(Line.split()[0] == '_atom_site_fract_z'):
                FindPos = True
                continue
        # Read cell
        elif(len(Line.split())== 2):
            if(Line.split()[0] == '_cell_length_a'):
                StruCell.append(float(Line.split()[1]))
            elif(Line.split()[0] == '_cell_length_b'):
                StruCell.append(float(Line.split()[1]))
            elif(Line.split()[0] == '_cell_length_c'):
                StruCell.append(float(Line.split()[1]))
            elif(Line.split()[0] == '_cell_angle_alpha'):
                StruCell.append(float(Line.split()[1]))
            elif(Line.split()[0] == '_cell_angle_beta'):
                StruCell.append(float(Line.split()[1]))
            elif(Line.split()[0] == '_cell_angle_gamma'):
                StruCell.append(float(Line.split()[1]))
        # Read position
        if(FindPos):
            StruSymbol.append(Line.split()[0])
            StruScalePos.append([float(Line.split()[i]) for i in range(1, 4)])
    # Get structure
    StruOne = Atoms(symbols = StruSymbol,
                 cell = StruCell,
                 scaled_positions = StruScalePos)
    return StruOne


def GetEnergyAtoms():
    # Get energy and number of atoms from abacus 'running_cell-relax.log' file
    Lines = OpenFile('running_cell-relax.log')
    for line in Lines:
        # Find number of atoms
        if line.find('TOTAL ATOM NUMBER') != -1: 
            NumberAtoms = int(line.split(' = ')[1])
        # Find total energy
        elif line.find('final etot is') != -1: 
            TotalEnergy = re.findall(r'[-+]?\d+\.?\d*[eE]?[-+]?\d+', line)
            TotalEnergy = float(TotalEnergy[0])
    return NumberAtoms, TotalEnergy


StruAll = list()
def ReadAbacus():
    for root, dirs, files in os.walk('.', topdown = True):
        for DirName in dirs:
            if(DirName[:4]=='file'):
                os.chdir(WorkPath + DirName + '/OUT.ABACUS')
                try:
                    StruAll.append(GetStruOne()) # Get structure
                    At, En = GetEnergyAtoms()    # Get number of atoms and energy
                    StruAll[-1].NumberAtoms = len(StruAll[-1].numbers)
                    StruAll[-1].Energy = float(En) / StruAll[-1].NumberAtoms
                    # StruAll[-1].Item = root + DirName
                    StruAll[-1].Item = DirName
                    StruAll[-1].fnm = DirName
                except:
                    if(len(StruAll) == 0):
                        continue
                    elif(StruAll[-1] == []):
                        del StruAll[-1]
                    elif(StruAll[-1].Energy == None):
                        del StruAll[-1]
    os.chdir(WorkPath)
    return StruAll
            

if __name__ == '__main__':
    StruAll = ReadAbacus()
    # Set results file
    ResultsDir = os.path.join(WorkPath, 'RunResults')
    if (not os.path.exists(ResultsDir)):
        os.mkdir(ResultsDir)
    os.chdir(ResultsDir)
    # Write result(structure, number of atoms, energy and density of energy)
    ResFile = open('Results.dat', 'w')
    ResFile.write('{0:<20s}{1:<24s}{2:<12s}{3:<24s}\n'.format('StructureName', 
                  'Energy(eV)',
                  'Number',
                  'DensityOfAtom(eV/atom)'))
    for stru in StruAll:
        # Write structure
        FileName = stru.Item + '_' + stru.get_chemical_formula()
        write(filename = FileName + '.cif' ,
              images = stru,
              format = 'cif') 
        # Write results
        ResFile.write('{0:<20s}{1:<24s}{2:<12s}{3:<24s}\n'.format(
                      FileName,
                      str(stru.Energy),
                      str(stru.NumberAtoms),
                      str(stru.Energy / stru.NumberAtoms)))
    ResFile.close()


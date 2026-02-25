import numpy as np
from pyscf import gto, scf
from pyscf import mcscf, mrpt

def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    num_atoms = int(lines[0].strip())
    atoms = []
    coordinates = []

    for line in lines[2:num_atoms + 2]:  
        parts = line.split()
        atoms.append(parts[0])
        coordinates.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return atoms, np.array(coordinates)

def run_nevpt2(atoms, coordinates, distance):
    mol = gto.Mole()
    mol.atom = "\n".join([f"{atom} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}" 
                         for atom, coord in zip(atoms, coordinates)])
    mol.basis = 'ccpvdz'
    mol.charge = 0
    mol.spin = 0
    mol.build()

    mf = scf.ROHF(mol)
    mf.kernel()

    mc = mcscf.CASSCF(mf, 2,2)
    mc = mc.state_average_([.50, .50])
    mc.fcisolver.nroots=2
    mc.fcisolver.spin=0
    mc.fix_spin_(ss=0)
    mc.kernel()
    
    orbital = mc.mo_coeff
    mc=mcscf.CASCI(mf,2,2)
    mc.fcisolver.nroots=2
    mc.fcisolver.spin=0
    mc.fix_spin_(ss=0)
    mc.kernel(orbital)
    
    e_corr0 = mrpt.NEVPT(mc,root=0).kernel()
    e_corr1 = mrpt.NEVPT(mc,root=1).kernel()
    e_tot0 = mc.e_tot[0] + e_corr0
    e_tot1 = mc.e_tot[1] + e_corr1

    return e_tot0, e_tot1

def main():
    distances = [3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 10.0]
    ct_energy = []
    for distance in distances:
        xyz_file = f"../00-xyz/long_CT_{int(distance*10)}.xyz"
        
        atoms, coordinates = read_xyz(xyz_file)

        e_state1, e_state2 = run_nevpt2(atoms, coordinates, distance)
        ct_energy.append(e_state2-e_state1)
    print(ct_energy)
if __name__ == "__main__":
    main()


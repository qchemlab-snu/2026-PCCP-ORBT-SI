import numpy as np
from pyscf import gto, scf
from pyscf import mcscf, mcpdft
from pyscf.sfnoci.sfnoci import SFGNOCI
from pyscf.sfnoci.gbci import GBDFT

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

def run_casci(atoms, coordinates, distance, functional):
    mol = gto.Mole()
    mol.atom = "\n".join([f"{atom} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}" 
                         for atom, coord in zip(atoms, coordinates)])
    mol.basis = 'ccpvdz'
    mol.charge = 0
    mol.spin = 0
    mol.build()

    mf = scf.ROHF(mol)
    mf.kernel()

    mc = mcpdft.CASCI(mf,functional ,2,2)
    mc = mc.state_average_([.50, .50])
    mc.fcisolver.nroots=2
    mc.fcisolver.spin=0
    mc.fix_spin_(ss=0)
    mc.kernel()
    
    return mc.e_states

def run_casscf(atoms, coordinates, distance, functional):
    mol = gto.Mole()
    mol.atom = "\n".join([f"{atom} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}" 
                         for atom, coord in zip(atoms, coordinates)])
    mol.basis = 'ccpvdz'
    mol.charge = 0
    mol.spin = 0
    mol.build()

    mf = scf.ROHF(mol)
    mf.kernel()

    mc = mcpdft.CASSCF(mf,functional,2,2)
    mc = mc.state_average_([.50, .50])
    mc.fcisolver.nroots=2
    mc.fcisolver.spin=0
    mc.fix_spin_(ss=0)
    mc.kernel()
    
    return mc.e_states

def run_gbdft(atoms, coordinates, distance,functional):
    mol = gto.Mole()
    mol.atom = "\n".join([f"{atom} {coord[0]:.10f} {coord[1]:.10f} {coord[2]:.10f}" 
                         for atom, coord in zip(atoms, coordinates)])
    mol.basis = 'ccpvdz'
    mol.charge = 0
    mol.spin = 0
    mol.build()

    mf = scf.ROHF(mol)
    mf.kernel()

    mygbci = GBDFT(mf,functional,2,(1, 1),nroots=2,groupA={'furan':[0,1,2,3,4,5,6,7,8],'dcne':[9,10,11,12,13,14,15,16]})
    mygbci.kernel(mf.mo_coeff)

    return mygbci.e_states

def main():
    functional = "tPBE" #or "tPBE0"
    distances = [3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 10.0]
    ct_energy_ci, ct_energy_scf, ct_energy_gb = [], [], []
    for distance in distances:
        xyz_file = f"../00-xyz/long_CT_{int(distance*10)}.xyz"
        
        atoms, coordinates = read_xyz(xyz_file)

        e_state1, e_state2 = run_casci(atoms, coordinates, distance, functional)
        ct_energy_ci.append(e_state2-e_state1)
        e_state1, e_state2 = run_casscf(atoms, coordinates, distance, functional)
        ct_energy_scf.append(e_state2-e_state1)
        e_state1, e_state2 = run_gbdft(atoms, coordinates, distance, functional)
        ct_energy_gb.append(e_state2-e_state1)
 
    print("CASCI+PDFT")
    print(ct_energy_ci)
    print("CASSCF+PDFT")
    print(ct_energy_scf)
    print("GBCI+PDFT")
    print(ct_energy_gb)
 
if __name__ == "__main__":
    main()


from pyscf import gto, scf, mcscf
from pyscf.mcscf import addons 

mol = gto.Mole()
mol.atom = '''
H 2.490675 0.939499 -0.000156
N 2.085280 0.000000 0.000003
H 2.490739 -0.469633 0.813668
H 2.490627 -0.469860 -0.813586
F -0.222963 0.000000 0.000001
F -1.710402 0.000000 0.000000

'''
mol.basis = 'ccpvdz'
mol.charge = 0
mol.spin = 0
mol.build()

mf = scf.ROHF(mol)
mf.kernel()

cas_list=[9, 14, 15]
nroots=4

mc_ci = mcscf.CASCI(mf, 3, 4)
mc_ci = mc_ci.state_average_([1/nroots for _ in range(nroots)])
mc_ci.fcisolver.spin=0
mc_ci.fix_spin_(ss=0)
mo = addons.sort_mo(mc_ci,mf.mo_coeff,cas_list,1)
mc_ci.kernel()

mc_scf = mcscf.CASSCF(mf, 3, 4)
mc_scf = mc_scf.state_average_([1/nroots for _ in range(nroots)])
mc_scf.fcisolver.spin=0
mc_scf.fix_spin_(ss=0)
mo = addons.sort_mo(mc_scf,mf.mo_coeff,cas_list,1)
mc_scf.kernel()

from pyscf.sfnoci.sfnoci import SFGNOCI
mc_gb = SFGNOCI(mf,3,(2, 2), groupA={'nh3': [0, 1, 2, 3], 'f2': [3, 4]})
mc_gb.fcisolver.nroots = 4
mc_gb.fcisolver.spin=0
mc_gb.fix_spin_(ss=0)
mo = addons.sort_mo(mc_gb, mf.mo_coeff, cas_list, 1)
e, _, ci = mc_gb.kernel(mo)
print(e)


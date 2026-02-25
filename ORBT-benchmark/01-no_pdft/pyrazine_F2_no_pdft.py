from pyscf import gto, scf, mcscf
from pyscf.mcscf import addons 

mol = gto.Mole()
mol.atom = '''
N 0.995753 1.433963 0.000000
C 0.995753 0.702984 1.137230
C 0.995753 -0.702984 1.137230
N 0.995753 -1.433963 0.000000
C 0.995753 -0.702984 -1.137230
C 0.995753 0.702984 -1.137230
H 0.995753 1.261352 2.081529
H 0.995753 -1.261351 2.081529
H 0.995753 -1.261352 -2.081529
H 0.995753 1.261351 -2.081529
F -2.097479 0.721626 0.000000
F -2.097479 -0.721626 0.000000
'''
mol.basis = 'ccpvdz'
mol.charge = 0
mol.spin = 0
mol.build()

mf = scf.ROHF(mol)
mf.kernel()

cas_list=[23, 29, 30, 31]
nroots=4

mc_ci = mcscf.CASCI(mf, 4, 6)
mc_ci = mc_ci.state_average_([1/nroots for _ in range(nroots)])
mc_ci.fcisolver.spin=0
mc_ci.fix_spin_(ss=0)
mo = addons.sort_mo(mc_ci,mf.mo_coeff,cas_list,1)
mc_ci.kernel()

mc_scf = mcscf.CASSCF(mf, 4, 6)
mc_scf = mc_scf.state_average_([1/nroots for _ in range(nroots)])
mc_scf.fcisolver.spin=0
mc_scf.fix_spin_(ss=0)
mo = addons.sort_mo(mc_scf,mf.mo_coeff,cas_list,1)
mc_scf.kernel()

from pyscf.sfnoci.sfnoci import SFGNOCI
mc_gb = SFGNOCI(mf,4,(3, 3), groupA={'pyr': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'f2': [10, 11]})
mc_gb.fcisolver.nroots = 4
mc_gb.fcisolver.spin=0
mc_gb.fix_spin_(ss=0)
mo = addons.sort_mo(mc_gb, mf.mo_coeff, cas_list, 1)
e, _, ci = mc_gb.kernel(mo)
print(e)


from pyscf import gto, scf, mcscf, mcpdft
from pyscf.mcscf import addons 
from pyscf.sfnoci.gbci import GBDFT
from pyscf.sfnoci.sfnoci import SFGNOCI

mol = gto.Mole()
mol.atom = '''
N -2.420290 -0.000734 -0.132257
H -1.815288 -0.803134 0.063560
H -1.818726 0.804803 0.061262
H -3.083243 0.003370 0.647931
O 0.983746 0.000141 -0.561705
F 0.656308 -1.133526 0.264727
F 0.655717 1.133681 0.264666
'''
mol.basis = 'ccpvdz'
mol.charge = 0
mol.spin = 0
mol.build()

mf = scf.ROHF(mol)
mf.kernel()

cas_list=[17, 18, 19]
nroots=4

mc_ci = mcpdft.CASCI(mf,'tPBE0', 3, 4)
mc_ci = mc_ci.state_average_([1/nroots for _ in range(nroots)])
mc_ci.fcisolver.spin=0
mc_ci.fix_spin_(ss=0)
mo = addons.sort_mo(mc_ci,mf.mo_coeff,cas_list,1)
mc_ci.kernel(mo)

mc_scf = mcpdft.CASSCF(mf,'tPBE0',3, 4)
mc_scf = mc_scf.state_average_([1/nroots for _ in range(nroots)])
mc_scf.fcisolver.spin=0
mc_scf.fix_spin_(ss=0)
mo = addons.sort_mo(mc_scf,mf.mo_coeff,cas_list,1)
mc_scf.kernel(mo)

mc_gb = GBDFT(mf,'tPBE0',3,(2, 2),nroots=4,groupA={'nh3': [0, 1, 2, 3], 'of2': [4, 5, 6]},spin=0)
mo = addons.sort_mo(mc_gb, mf.mo_coeff, cas_list, 1)
mc_gb.kernel(mo)
print(mc_gb.e_states)


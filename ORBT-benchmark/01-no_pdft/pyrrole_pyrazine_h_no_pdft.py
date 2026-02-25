from pyscf import gto, scf, mcscf
from pyscf.mcscf import addons 

mol = gto.Mole()
mol.atom = '''
N -1.116277 -0.000150 0.019717
C -1.840966 1.141214 0.010095
C -3.246427 1.141801 -0.008562
N -3.972069 0.001043 -0.018193
C -3.248391 -1.140963 -0.008585
C -1.842932 -1.142795 0.010072
H -1.279558 2.083082 0.017546
H -3.807049 2.084137 -0.016006
H -3.810634 -2.082333 -0.016048
H -1.282199 -2.085066 0.017517
N 1.906183 -0.000072 0.015242
C 2.699989 1.126323 0.003913
C 2.700047 -1.126427 0.003917
C 4.034805 0.715035 -0.015132
C 4.034842 -0.715070 -0.015130
H 0.879481 -0.000601 0.029892
H 2.260267 2.123953 0.010185
H 2.260376 -2.124079 0.010193
H 4.904166 1.372849 -0.027538
H 4.904236 -1.372840 -0.027534

'''
mol.basis = 'ccpvdz'
mol.charge = 0
mol.spin = 0
mol.build()

mf = scf.ROHF(mol)
mf.kernel()

cas_list=[39, 40, 41]
nroots=4

mc_ci = mcscf.CASCI(mf, 3, 2)
mc_ci = mc_ci.state_average_([1/nroots for _ in range(nroots)])
mc_ci.fcisolver.spin=0
mc_ci.fix_spin_(ss=0)
mo = addons.sort_mo(mc_ci,mf.mo_coeff,cas_list,1)
mc_ci.kernel()

mc_scf = mcscf.CASSCF(mf, 3, 2)
mc_scf = mc_scf.state_average_([1/nroots for _ in range(nroots)])
mc_scf.fcisolver.spin=0
mc_scf.fix_spin_(ss=0)
mo = addons.sort_mo(mc_scf,mf.mo_coeff,cas_list,1)
mc_scf.kernel()

from pyscf.sfnoci.sfnoci import SFGNOCI
mc_gb = SFGNOCI(mf,3,(1, 1), groupA={'pyrr': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 'pyr': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
mc_gb.fcisolver.nroots = 4
mc_gb.fcisolver.spin=0
mc_gb.fix_spin_(ss=0)
mo = addons.sort_mo(mc_gb, mf.mo_coeff, cas_list, 1)
e, _, ci = mc_gb.kernel(mo)
print(e)


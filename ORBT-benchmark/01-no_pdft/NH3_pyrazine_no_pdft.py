from pyscf import gto, scf, mcscf
from pyscf.mcscf import addons 

mol = gto.Mole()
mol.atom = '''
N 0.551070 1.435370 0.000026
C 0.551176 0.702702 1.134927
C 0.551384 -0.702642 1.134899
N 0.551493 -1.435265 -0.000032
C 0.551386 -0.702589 -1.134927
C 0.551179 0.702756 -1.134889
H 0.551093 1.259441 2.080741
H 0.551464 -1.259418 2.080691
H 0.551470 -1.259320 -2.080746
H 0.551098 1.259531 -2.080682
N -2.516082 -0.000399 -0.000002
H -2.940485 0.466335 -0.807243
H -2.940280 -0.932952 0.000820
H -2.940487 0.467759 0.806414

'''
mol.basis = 'ccpvdz'
mol.charge = 0
mol.spin = 0
mol.build()

mf = scf.ROHF(mol)
mf.kernel()

cas_list=[23, 27, 29]
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
mc_gb = SFGNOCI(mf,3,(1, 1), groupA={'nh3': [10, 11, 12, 13], 'pyr': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
mc_gb.fcisolver.nroots = 4
mc_gb.fcisolver.spin=0
mc_gb.fix_spin_(ss=0)
mo = addons.sort_mo(mc_gb, mf.mo_coeff, cas_list, 1)
e, _, ci = mc_gb.kernel(mo)
print(e)


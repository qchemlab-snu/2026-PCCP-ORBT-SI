from pyscf import gto, scf, mcscf, mcpdft
from pyscf.mcscf import addons 
from pyscf.sfnoci.gbci import GBDFT
from pyscf.sfnoci.sfnoci import SFGNOCI

mol = gto.Mole()
mol.atom = '''
H 2.068365 -1.554770 -1.258236
H -2.068564 -1.553940 -1.258236
H 2.068365 -1.554770 1.258236
H -2.068564 -1.553940 1.258236
C 1.135821 -1.554583 -0.697884
C -1.136020 -1.554127 -0.697884
C 1.135821 -1.554583 0.697884
C -1.136020 -1.554127 0.697884
N -0.000100 -1.554355 -1.417402
N -0.000100 -1.554355 1.417402
H -0.770304 1.855801 -2.114612
H -0.770304 1.855801 2.114612
H 1.850809 1.855275 -1.358585
H 1.850809 1.855275 1.358585
H -2.130086 1.856074 0.000000
C -0.333285 1.855713 -1.125828
C -0.333285 1.855713 1.125828
C 0.985374 1.855448 -0.709235
C 0.985374 1.855448 0.709235
N -1.119278 1.855871 0.000000
'''
mol.basis = 'ccpvdz'
mol.charge = 0
mol.spin = 0
mol.build()

mf = scf.ROHF(mol)
mf.kernel()

cas_list=[38, 39, 40, 41]
nroots=4

mc_ci = mcpdft.CASCI(mf,'tPBE0', 4, 4)
mc_ci = mc_ci.state_average_([1/nroots for _ in range(nroots)])
mc_ci.fcisolver.spin=0
mc_ci.fix_spin_(ss=0)
mo = addons.sort_mo(mc_ci,mf.mo_coeff,cas_list,1)
mc_ci.kernel(mo)

mc_scf = mcpdft.CASSCF(mf,'tPBE0',4, 4)
mc_scf = mc_scf.state_average_([1/nroots for _ in range(nroots)])
mc_scf.fcisolver.spin=0
mc_scf.fix_spin_(ss=0)
mo = addons.sort_mo(mc_scf,mf.mo_coeff,cas_list,1)
mc_scf.kernel(mo)

mc_gb = GBDFT(mf,'tPBE0',4,(2, 2),nroots=4,groupA={'pyr': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 'pyrr': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},spin=0)
mo = addons.sort_mo(mc_gb, mf.mo_coeff, cas_list, 1)
mc_gb.kernel(mo)
print(mc_gb.e_states)


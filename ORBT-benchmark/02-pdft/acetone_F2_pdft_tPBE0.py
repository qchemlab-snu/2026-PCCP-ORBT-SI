from pyscf import gto, scf, mcscf, mcpdft
from pyscf.mcscf import addons 
from pyscf.sfnoci.gbci import GBDFT
from pyscf.sfnoci.sfnoci import SFGNOCI

mol = gto.Mole()
mol.atom = '''
H -2.543489 0.848415 -1.328942
C -1.483454 0.544023 -1.289081
H -0.874751 1.463810 -1.326362
H -1.243732 -0.093252 -2.152254
C -1.199386 -0.211141 0.000000
O -0.766094 -1.362999 0.000000
C -1.483454 0.544023 1.289082
H -1.243732 -0.093253 2.152254
H -0.874757 1.463813 1.326367
H -2.543491 0.848408 1.328938
F 1.631766 0.853847 0.000000
F 2.139396 -0.495631 0.000000
'''
mol.basis = 'ccpvdz'
mol.charge = 0
mol.spin = 0
mol.build()

mf = scf.ROHF(mol)
mf.kernel()

cas_list=[23, 24, 25, 26]
nroots=4

mc_ci = mcpdft.CASCI(mf,'tPBE0', 4, 6)
mc_ci = mc_ci.state_average_([1/nroots for _ in range(nroots)])
mc_ci.fcisolver.spin=0
mc_ci.fix_spin_(ss=0)
mo = addons.sort_mo(mc_ci,mf.mo_coeff,cas_list,1)
mc_ci.kernel(mo)

mc_scf = mcpdft.CASSCF(mf,'tPBE0',4, 6)
mc_scf = mc_scf.state_average_([1/nroots for _ in range(nroots)])
mc_scf.fcisolver.spin=0
mc_scf.fix_spin_(ss=0)
mo = addons.sort_mo(mc_scf,mf.mo_coeff,cas_list,1)
mc_scf.kernel(mo)

mc_gb = GBDFT(mf,'tPBE0',4,(3, 3),nroots=4,groupA={'acetone': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'f2': [10, 11]},spin=0)
mo = addons.sort_mo(mc_gb, mf.mo_coeff, cas_list, 1)
mc_gb.kernel(mo)
print(mc_gb.e_states)


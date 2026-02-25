from pyscf import gto, scf, mcscf, mcpdft
from pyscf.mcscf import addons 
from pyscf.sfnoci.gbci import GBDFT
from pyscf.sfnoci.sfnoci import SFGNOCI

mol = gto.Mole()
mol.atom = '''
H 1.811829 0.246785 1.837687
C 2.341053 0.546375 0.917315
H 3.348706 0.098100 0.941832
H 2.413351 1.642596 0.878699
C 1.569262 0.038569 -0.286392
O 1.118023 0.805512 -1.143722
C 1.383099 -1.464592 -0.377435
H 0.813573 -1.715706 -1.283526
H 2.363599 -1.969922 -0.390703
H 0.830656 -1.814945 0.510807
C -1.970874 1.149505 -0.504785
H -1.152760 1.388024 -1.196400
H -2.893231 0.879281 -1.034661
H -2.130317 1.966927 0.210009
N -1.514242 -0.049172 0.261983
O -0.842687 0.159540 1.295692
O -1.782949 -1.169899 -0.222646
'''
mol.basis = 'ccpvdz'
mol.charge = 0
mol.spin = 0
mol.build()

mf = scf.ROHF(mol)
mf.kernel()

cas_list=[31, 32, 33, 34]
nroots=4

mc_ci = mcpdft.CASCI(mf,'tPBE', 4, 4)
mc_ci = mc_ci.state_average_([1/nroots for _ in range(nroots)])
mc_ci.fcisolver.spin=0
mc_ci.fix_spin_(ss=0)
mo = addons.sort_mo(mc_ci,mf.mo_coeff,cas_list,1)
mc_ci.kernel(mo)

mc_scf = mcpdft.CASSCF(mf,'tPBE',4, 4)
mc_scf = mc_scf.state_average_([1/nroots for _ in range(nroots)])
mc_scf.fcisolver.spin=0
mc_scf.fix_spin_(ss=0)
mo = addons.sort_mo(mc_scf,mf.mo_coeff,cas_list,1)
mc_scf.kernel(mo)

mc_gb = GBDFT(mf,'tPBE',4,(2, 2),nroots=4,groupA={'acetone': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 'ch3no2': [10, 11, 12, 13, 14, 15, 16]},spin=0)
mo = addons.sort_mo(mc_gb, mf.mo_coeff, cas_list, 1)
mc_gb.kernel(mo)
print(mc_gb.e_states)


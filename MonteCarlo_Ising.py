import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from numba import int32,float32
from numba.experimental import jitclass
import secrets as scr
# random gennrerator using random bit
entropy=eval(f"0x{scr.randbits(128):x}")
rng=np.random.default_rng(entropy)

spec=[
    ('matrix',float32[:,:]),
    ('Lx',int32),
    ('Ly',int32),
]
#
#####################################################################
# Define a class 'State' to describe the configuration of the system and function to evaluate 
# the energy and magnetization (total magnetic moment divided by number of spins)
# state is represented by a matrix of Ly row and Lx column matrix with elements 1. and -1. 
# correspond to the possible value of spin on each site (Classical not Quantum)
# the Hamiltonian is given in jupyter notebook.
#
# The Metropolis procedure and simulate annealing procedure differ in whether the temperature
# is fixed
#####################################################################
#
@jitclass(spec)
class State:
    def __init__(self, statemat):
        self.matrix=statemat
        self.Lx=int32(statemat.shape[1])
        self.Ly=int32(statemat.shape[0])

    def _getEnergy(self,J_int:float,h_ext:float)->float:
        Energy=0.
        for i in range(self.Ly):
            for j in range(self.Lx):
                site=self.matrix[i,j]
                Energy+=neigborEnergy(self,i,j,site,J_int)/2 -h_ext*site
        return Energy
    def _getMoment(self)->float:
        return np.sum(self.matrix)
# self energy of one spin and its share in the exchange interaction with its neighboring spins  
@jit(nopython=True)
def neigborEnergy(state:State,i:int,j:int, site:float, J_int:float)->float:
    up,down=((i-1)%state.Ly,j),((i+1)%state.Ly,j)
    left,right=(i,(j-1)%state.Lx),(i,(j+1)%state.Lx)
    neigbors=[up,down,left,right]
    energy= 0.0
    for ngb in neigbors:
        if state.matrix[ngb]==site:
            energy+=J_int 
        else:
            energy-=J_int
    return energy

# simulate Annealing algporith with Temp=(Ti,Tf) temperature from Ti cooling down to Tf in 1/(a+blog(t))
# style with markove chain length MkvL. display is set to default None, if designated as an int, 
# it represent the number of frames of the animation
@jit(nopython=True)    
def SimAnnealing(state:State,Temp:tuple,idxs,jdxs,probs,MkvL:int,J_int:float,h_ext:float,display=None):
    kb=1.0  
    slope=(Temp[0]/Temp[1]-1.)/np.log(MkvL)
    betas=(1.+slope*np.log(1+np.arange(MkvL)))/kb
    if display is not None:
        step=int(MkvL/display)
        pics=[state.matrix]
    for t in range(MkvL):
        i,j=idxs[t],jdxs[t]
        prob=probs[t]
        site=state.matrix[i,j]
        energy_ini=neigborEnergy(state,i,j,site,J_int)-h_ext*site # the neighboring enegy under current state
        energy_fin=neigborEnergy(state,i,j,-site,J_int)+h_ext*site # neighboring energy if spin (i,j) flipped
        delta=energy_fin-energy_ini # the energy shift if spin(i,j) shifted
        accept_prob=np.exp(min(0.,-betas[t]*delta))
        if prob<accept_prob: #if E_fin<E_ini the spin(i,j) should be surely flipped else it
            state.matrix[i,j]=-state.matrix[i,j] # should be flipped with probability exp(-(Ef-Ei)/kT)
        if (display is not None) and t%step==0: # for the need of displaying animation
            pics.append(state.matrix.copy())
    if display is not None:
        return pics
# similar to Simulate Annealing algorithm with constant temperature Temp
@jit(nopython=True)    
def Metropolis(state:State,Temp:float,idxs,jdxs,probs,MkvL:int,J_int:float,h_ext:float,display=None):
    kb=1.0
    beta=1./(kb*Temp)
    if display is not None:
        step=int(MkvL/display)
        pics=[state.matrix]
    for t in range(MkvL):
        i,j=idxs[t],jdxs[t]
        prob=probs[t]
        site=state.matrix[i,j]
        energy_ini=neigborEnergy(state,i,j,site,J_int)-h_ext*site
        energy_fin=neigborEnergy(state,i,j,-site,J_int)+h_ext*site
        delta=energy_fin-energy_ini
        accept_prob=np.exp(min(0.,-beta*delta))
        if prob < accept_prob:
            state.matrix[i,j]=-state.matrix[i,j]
        if (display is not None) and t%step==0:
            pics.append(state.matrix.copy())
    if display is not None:
        return pics
    
# One round of  Markov Chain Monte Carlo(MCMC) procedure, method is set default to 'MC', i.e. Metropolis.
# if MkvL not designated, it picks the number of 5 times the volume of system.
# J_int default -1. (FM), h_ext default 0. (zero external field)
def getsample(Lx:int,Ly:int,T_ini:float,J_int:float,h_ext:float,initial='random',method='MC',MkvL=None):
    MkvL= 5*Lx*Ly if MkvL is None else MkvL
    if initial=='random':
        state=State(rng.choice([float32(1.),float32(-1.)],size=(Ly,Lx))) # randomly chosen initial state
    elif initial=='unif':
        state=State(np.ones((Ly,Lx),dtype='float32'))   # uniform initial state
    #  choosing random values early for the numba.jit support of random generator module is NOT complete.
    rdxsi=rng.integers(0,Ly,size=MkvL) # random i-index of spins
    rdxsj=rng.integers(0,Lx,size=MkvL) # random j-index of spins 
    rdprob=rng.uniform(0.,1.,size=MkvL) # random probability metric of flipping spins
    if method=='MC':
        Metropolis(state,T_ini,rdxsi,rdxsj,rdprob,MkvL,J_int,h_ext)
    elif method=='SA':
        SimAnnealing(state,T_ini,rdxsi,rdxsj,rdprob,MkvL,J_int,h_ext)
    else:
        print('wait for later development.')
    Energy=state._getEnergy(J_int,h_ext)
    Moment=state._getMoment()
    return Energy,Moment
# Display one round of MCMC procedure
def DisplayOneRound(Lx:int,Ly:int,T_ini:float,J_int:float,h_ext:float,initial='random',method='MC',n_shots:int=50,MkvL=None):
    MkvL= 5*Lx*Ly if MkvL is None else MkvL
    if initial=='random':
        state=State(rng.choice([float32(1.),float32(-1.)],size=(Ly,Lx)))
    elif initial=='uniform':
        state=State(np.ones((Ly,Lx),dtype='float32'))   # uniform initial state
    rdxsi=rng.integers(0,Ly,size=MkvL)
    rdxsj=rng.integers(0,Lx,size=MkvL)
    rdprob=rng.uniform(0.,1.,size=MkvL)
    fig,ax=plt.subplots(figsize=(6,6))
    ims=[]
    if method=='MC':
        pics=Metropolis(state,T_ini,rdxsi,rdxsj,rdprob,MkvL,J_int,h_ext,display=n_shots)
    elif method=='SA':
        pics=SimAnnealing(state,T_ini,rdxsi,rdxsj,rdprob,MkvL,J_int,h_ext,display=n_shots)
    else:
        print('wait for later development.')
    for k in range(len(pics)):
        if k==0:
            ax.pcolor(pics[0],edgecolors='k',vmin=-1.,vmax=1.) # using Axes.matshow() to visualize different state. There are better methods 
        else:
            im=ax.pcolor(pics[k],edgecolors='k',vmin=-1.,vmax=1.,animated=True)
            ims.append([im])
    #ani=anm.ArtistAnimation(fig,ims,blit=True) # if using python file only rather than jupyter, 
    #plt.show()                                 # delete the '#' of these two lines and the return line
    return fig,ims

class Result:  # define a class result to save averaged results of may round of MCMC
    def __init__(self,energy:float,moment:float,varsq:float):
        self.energy=energy
        self.moment=moment
        self.varsq=varsq

# Averaging the results with N_sample round of MCMC
def Mean_result(Lx:int,Ly:int,T_ini:float,N_sample:int,J_int,h_ext,initial='random',method='MC',MkvL=None):
    total_energy,total_moment,total_square=0.,0.,0.
    for i in range(N_sample):
        energy,moment=getsample(Lx,Ly,T_ini,J_int,h_ext,initial,method,MkvL)
        total_energy+=energy
        total_moment+=moment
        total_square+=energy*energy
    mean_energy=total_energy/N_sample
    mean_moment=total_moment/N_sample
    varsq_energy=total_square/N_sample-mean_energy*mean_energy
    result=Result(mean_energy/(Lx*Ly),mean_moment/(Lx*Ly),varsq_energy)
    return result

#if __name__=='__main__':
#    DisplayOneRound(40,40,1.e-3,n_shots=50,MkvL=10000)
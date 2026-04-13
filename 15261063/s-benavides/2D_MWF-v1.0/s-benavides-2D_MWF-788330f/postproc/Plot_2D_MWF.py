import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import h5py
import matplotlib.cm as cm
from importlib import import_module
import sys
sys.path.append('../')

plt.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.serif"] = "Times New Roman"

def sortKeyFunc(s):
    s = s.split('_')[-1] # s{num}.h5
    s = s[1:-3] #num
    return int(s)

def read_dedalus(folder,intype='snapshots'):
    if intype=='snapshots':
        data = []
        ss = sorted(glob.glob(folder+'/'+intype+'/'+intype+'*.h5'),key=sortKeyFunc)
        for file in ss:
            data.append(h5py.File(file,'r'))
    elif intype=='time_series':
        data = dict([])
        ss = sorted(glob.glob(folder+'/'+intype+'/'+intype+'*.h5'),key=sortKeyFunc)
        for ii,file in enumerate(ss):
            if ii==0:
                data['time'] = h5py.File(file,'r')['scales']['sim_time']
                data['timestep'] = h5py.File(file,'r')['scales']['iteration']
            else:
                data['time'] = np.hstack((data['time'],h5py.File(file,'r')['scales']['sim_time']))
                data['timestep'] = h5py.File(file,'r')['scales']['iteration']

        temp = h5py.File(ss[0],'r')
        tasks = list(temp['tasks'].keys())
        for task in tasks:
            for ii,file in enumerate(ss):
                if ii==0:
                    data[task] = h5py.File(file,'r')['tasks'][task][:,0,0]
                else:
                    data[task] = np.hstack((data[task],h5py.File(file,'r')['tasks'][task][:,0,0]))
        data['len_ss'] = len(ss)
    return data

def snapshot_slice(n,data,field):
    for ii,subdat in enumerate(data):
        if np.any(subdat['scales']['write_number'][:]==n):
            n_m = np.squeeze(np.where(subdat['scales']['write_number'][:]==n))
            s = ii
    return [s,n_m]

#############
### Load data
# Choose input directory
idir = '../'

# Choose output directory
odir = './'

# Searches through all directories in 'Data' folder (which are named after experiments) and imports the data:
dirs = sorted(glob.glob(idir+'Lx224*tube*minimal*'))

runs = []
Res = []
for file in dirs:
    run = file.split('/')[1]

    # Res
    Re = float(run.split('_')[1][2:].replace('d','.'))
    Res.append(Re)
    runs.append(run)
    
    
Res = np.array(Res)
runs = np.array(runs)
argRes = Res.argsort()
Res = Res[argRes]
runs = runs[argRes]
Res = np.unique(Res)
print(runs)
print(Res)

# Import the data for each run
data = dict([])
for run in runs:
    data[run] = read_dedalus(idir+run)

t_data = dict([])
for run in runs:
    try:
        print(run)
        t_data[run] = read_dedalus(idir+run,intype='time_series')
    except Exception as e:
        print(run,"NO DATA")
        print(e)
      
params_data = dict([])
for run in runs:
    file = str(run+'/params').replace('/','.')
    params = import_module(file)
    params_data[run] = params

# Let's see what is in our imported variables.
# We saved the data in what's called a 'dictionary', which names its variables with strings. To see the experiments do the following:
for run in runs:
    print(run)
    print(data[run])
    for subdat in data[run]:
        print(subdat['scales'].keys())
        print(subdat['tasks'].keys())

###############
### Time series
# Choose run
lw  = 2.5
alpha = 1
Res_temp = np.copy(Res)
minscale = np.nan
maxscale = np.nan
meansu = []
meansq = []
alist = []
stds = []
for ii,run in enumerate(t_data): 
    Re = float(run.split('_')[1][2:].replace('d','.'))
    ls = '-'
    label='Re = %s' % int(Re)
    
    time = t_data[run]['time'][:]
    KE = t_data[run]['en_ls'][:]
    plt.figure(1,figsize=(10,6))
    plt.plot(time,KE,marker='',label = run)
    
    KE = t_data[run]['q0'][:]
    plt.figure(2,figsize=(10,6))
    plt.plot(time,KE,marker='',label = run)
    
    KE = t_data[run]['q1'][:]
    plt.figure(3,figsize=(10,6))
    plt.plot(time,KE,marker='',label = run)#,c=((Re-np.min(Res))/(np.max(Res)-np.min(Res)),0,0,1),lw=lw,ls = ls,alpha=alpha,label=Re)
            
log_yax = False

plt.figure(1,figsize=(10,6))
plt.xlabel("time")
plt.ylabel("KE_ls")
plt.legend(fontsize=15,loc=(1.01,0.0))
if log_yax:
    plt.gca().set_yscale('log')
plt.tight_layout()

plt.figure(2,figsize=(10,6))
plt.xlabel("time")
plt.ylabel("q0")
plt.legend(fontsize=15,loc=(1.01,0.0))
if log_yax:
    plt.gca().set_yscale('log')
plt.tight_layout()

plt.figure(3,figsize=(10,6))
plt.xlabel("time")
plt.ylabel("q1")
plt.legend(fontsize=15,loc=(1.01,0.0))
if log_yax:
    plt.gca().set_yscale('log')
plt.tight_layout()

plt.show()

#############
### Snapshots
runs_snaps = np.copy(runs)
plt.figure(figsize=(8,6))
for run in runs_snaps[:]:
    Re = float(run.split('_')[1][2:].replace('d','.'))
    n_max = data[run][-1]['scales']['write_number'][-1] # n_max is frame number, but with python we have to subtract one
    n = np.min([n_max,n_max])
    print("n_max = %s" % n_max)
    s,n = snapshot_slice(n,data[run],'u0')

    # Read info:
    time = data[run][s]['scales']['sim_time'][n]

    # Define array for x and z axes
    X = data[run][s]['scales']['x']['1.0'][:]
    Z = data[run][s]['scales']['z']['1.0'][:]
    Lx = np.max(X)
    Lz = np.max(Z)

    size = 10
    plt.figure(figsize=(10*3/2,6))#2*size,size*(H/L)))

    u0 = data[run][s]['tasks']['u0'][n,:,:]
    w0 = data[run][s]['tasks']['w0'][n,:,:]
    q0 = data[run][s]['tasks']['q0'][n,:,:]

    # Plot the turbulence
    Re = float(run.split('_')[1][2:].replace('d','.'))
    plt.pcolormesh(X,Z,q0.T,vmin=0,vmax=np.max(q0[:]),cmap= cm.copper)
    plt.colorbar(label=r'$q_0$')

    # Now the flow field
    from scipy.interpolate import interp2d

    # regularly spaced grid spanning the domain of x and y 
    Xi = np.linspace(X.min(), X.max(), int(X.size/2))
    Zi = np.linspace(Z.min(), Z.max(), int(Z.size/2))

    # bicubic interpolation
    ru = interp2d(X,Z, u0.T)(Xi, Zi)
    rw = interp2d(X,Z, w0.T)(Xi, Zi)

    plt.streamplot(Xi, Zi, ru, rw,color='w',density = [0.5,0.5])#,density =[0.4, 0.4],arrowsize=0.0,minlength=0.3)

    plt.xlabel(r"$x$")
    plt.ylabel(r"$z$")
    plt.xlim(0,Lx)
    plt.ylim(0,Lz)
    plt.gca().set_aspect(1)
    plt.title(run+r' time = %f' % time,fontsize=15)
    plt.show()

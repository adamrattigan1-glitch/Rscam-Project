import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation,rc
import glob as glob
import h5py
import imageio
import sys

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


# Choose input directory
idir = '../'

# Choose output directory
odir = './'

# Choose run
#run = 'Lx224Lz100_Re90_gauss_april_v6d7'
run = str(sys.argv[1])
# Plot with pcolormesh?
mesh = True
#mesh = False

temp = run.split('_')[0]
temp = temp.split('L')
Lx = int(temp[1][1:])
Lz = int(temp[2][1:])

# Load data
data = dict([])
data[run] = read_dedalus(idir+run)

# Total number of frames
frame_max = data[run][-1]['scales']['write_number'][-1] 

# Define array for x and z axes
X = data[run][0]['scales']['x']['1.0'][:]
Z = data[run][0]['scales']['z']['1.0'][:]
#Lx = np.max(X)
#Lz = np.max(Z)

Re = float(run.split('_')[1][2:].replace('d','.'))

#Field
field = 'q0'

leg = {
    'u0':r"$u_0$",
    'u1':r"$u_1$",
    'v1':r"$v_1$",
    'w0':r"$w_0$",
    'w1':r"$w_1$",
    'q0':r"$q_0$",
    'q1':r"$q_1$",
}

# Find max{|T'|} to make proper, symmetric colorbar:
cbarlim = np.nanmax(np.abs(np.transpose(data[run][-1]['tasks'][field][:,:,:])))

# Plot!
size = 10

# Choose your frames per second:
length=12 #seconds
frames_per_second=np.min([int(frame_max/length),30])
if mesh:
    writer = imageio.get_writer(odir+'/'+run+'_'+field+'_mesh.mp4', mode="I", codec='h264',bitrate=1800000,fps = frames_per_second,output_params=['-s','%sx750' % int(round(750*(Lx/Lz)))])
else:
    writer = imageio.get_writer(odir+'/'+run+'.mp4', mode="I", codec='h264',bitrate=1800000,fps = frames_per_second,output_params=['-s','%sx750' % int(round(750*(Lx/Lz)))])


# Reads binary files
for frame in range(frame_max):
    print("Working on frame %s of %s" % (frame+1,frame_max))
    fig = plt.figure(1,figsize=(size,size*(Lz/Lx)))
    s,n = snapshot_slice(frame+1,data[run],field)
    if mesh:
        plt.pcolormesh(X,Z,np.transpose(data[run][s]['tasks'][field][n,:,:]),vmin=0,vmax=cbarlim,cmap= 'copper')
        #plt.pcolormesh(X,Z,np.transpose(data[run][s]['tasks'][field][n,:,:]),vmin=-cbarlim,vmax=cbarlim,cmap= 'bwr')
    else:
        plt.contourf(X,Z,np.transpose(data[run][s]['tasks'][field][n,:,:]),vmin=0,vmax=cbarlim,cmap= 'copper')
    cb=plt.colorbar()
    cb.set_label(leg[field])
    time = data[run][s]['scales']['sim_time'][n]
    plt.title(run+'\n'+r'$Re =$ %i, time = %f' % (Re,time),fontsize=15)
    plt.xlabel('Streamwise')
    plt.ylabel('Spanwise')
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    writer.append_data(img)
    plt.close()

print("done making %s for run %s" % (field,run))

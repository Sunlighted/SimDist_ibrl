import h5py, numpy as np, glob
A=[0,0,0,0.0825,-0.0825,0,0.088,0]; Dd=[0.333,0,0.316,0,0.384,0,0,0.107]; AL=[0,-np.pi/2,np.pi/2,np.pi/2,-np.pi/2,np.pi/2,np.pi/2,0]
TCP=0.17  # flange -> fingertip-ish along flange z (Robotiq 2F-85 + mount), only for mm-scale differences
def fk(q):
    T=np.eye(4)
    for i in range(8):
        th=q[i] if i<7 else 0; a,d,al=A[i],Dd[i],AL[i]
        ca,sa,ct,st=np.cos(al),np.sin(al),np.cos(th),np.sin(th)
        T=T@np.array([[ct,-st,0,a],[st*ca,ct*ca,-sa,-sa*d],[st*sa,ct*sa,ca,ca*d],[0,0,0,1]])
    return (T@np.array([0,0,TCP,1]))[:3]
import sys; D=sys.argv[1].rstrip('/')+'/'
allv=[];alls=[]
for fn in sorted(glob.glob(D+'*.h5')):
    f=h5py.File(fn,'r'); c=f['action/joint_positions'][:]; q=f['observation/joint_positions'][:]; g=f['observation/gripper_knuckle'][:]
    pq=np.array([fk(x) for x in q]); pc=np.array([fk(x) for x in c])
    v=np.linalg.norm(np.diff(pq,axis=0),axis=1)/0.1; vz=np.diff(pq[:,2])/0.1
    # static sag: frames where commanded TCP moves < 5 mm/s for 3 frames
    vc=np.linalg.norm(np.diff(pc,axis=0),axis=1)/0.1
    slow=np.where(np.convolve(vc<0.005,np.ones(3),'same')==3)[0]
    sag=(pq-pc)[slow]
    # lag in TCP: fit shift k frames
    lagk=min(range(0,5),key=lambda k: np.mean(np.linalg.norm(pq[k:]-pc[:len(pc)-k],axis=1)))
    zmin=pq[:,2].min(); near=np.where(pq[:-1,2]<zmin+0.03)[0]
    allv+=list(v); 
    print(fn.split('/')[-1], f'TCP speed p50 {100*np.median(v):.1f} p90 {100*np.percentile(v,90):.1f} p99 {100*np.percentile(v,99):.1f} max {100*v.max():.1f} cm/s | >8cm/s {100*np.mean(v>0.08):.0f}% of steps',
          f'| cmd-meas TCP dist p50 {1e3*np.median(np.linalg.norm(pq-pc,axis=1)):.1f} max {1e3*np.linalg.norm(pq-pc,axis=1).max():.1f} mm, best shift {lagk} frames',
          f'| static (meas-cmd) n={len(slow)} xyz mm {np.round(1e3*np.median(sag,0),1)}',
          f'| last-3cm descent p90 {100*np.percentile(np.maximum(-vz[near],0),90):.1f} cm/s')
allv=np.array(allv); print(f'ALL: p50 {100*np.median(allv):.1f} p90 {100*np.percentile(allv,90):.1f} p99 {100*np.percentile(allv,99):.1f} cm/s; >8 cm/s {100*np.mean(allv>0.08):.0f}%  >5 {100*np.mean(allv>0.05):.0f}%')
print('\n-- regress (meas - cmd) TCP = bias + lag * (-cmd velocity), per axis, all demos pooled')
E=[];V=[]
for fn in sorted(glob.glob(D+'*.h5')):
    f=h5py.File(fn,'r'); c=f['action/joint_positions'][:]; q=f['observation/joint_positions'][:]
    pq=np.array([fk(x) for x in q]); pc=np.array([fk(x) for x in c])
    E.append((pq-pc)[1:-1]); V.append((pc[2:]-pc[:-2])/0.2)
E=np.concatenate(E); V=np.concatenate(V)
for i,ax in enumerate('xyz'):
    X=np.c_[np.ones(len(V)),-V[:,i]]; b,res,_,_=np.linalg.lstsq(X,E[:,i],rcond=None)
    r=E[:,i]-X@b
    print(f' {ax}: bias {1e3*b[0]:+.1f} mm, lag {1e3*b[1]:.0f} ms, resid sd {1e3*r.std():.1f} mm (raw sd {1e3*E[:,i].std():.1f})')

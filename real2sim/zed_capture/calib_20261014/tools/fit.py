import sys,os,glob,json
sys.path.insert(0,os.path.dirname(__file__))
from geo import *
from scipy.optimize import minimize
BOXES=[((0,-0.02,0),(0.005,0.06,0.005)),((0,0.06,0),(0.02,0.02,0.02))]
def box_pts(c,h,n=6):
    g=np.linspace(-1,1,n); P=[]
    for a in g:
        for b in g:
            for s in(-1,1):
                P+= [(s,a,b),(a,s,b),(a,b,s)]
    return np.array(P)*h+c
BP=[box_pts(np.array(c),np.array(h)) for c,h in BOXES]
def render(pose,proj,shape,zt=ZT):
    x,y,yaw=pose; Rz=R.from_euler("z",yaw,degrees=True).as_matrix(); o=np.array([x,y,zt+0.02])
    m=np.zeros(shape,np.uint8)
    for P in BP:
        uv=proj((Rz@P.T).T+o)
        if uv is None: continue
        cv2.fillConvexPoly(m,cv2.convexHull(uv.astype(np.float32)).astype(np.int32),1)
    return m
def redmask(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    return (((hsv[...,0]<10)|(hsv[...,0]>168))&(hsv[...,1]>110)&(hsv[...,2]>40)).astype(np.uint8)
def fit(img,proj,init,roi_pad=60,zt=ZT):
    red=redmask(img)
    m0=render(init,proj,red.shape,zt)
    ys,xs=np.nonzero(cv2.dilate(m0,np.ones((roi_pad,roi_pad),np.uint8)))
    roi=np.zeros_like(red); roi[ys.min():ys.max()+1,xs.min():xs.max()+1]=1
    # keep only red components that overlap the ROI and are not huge (tripod)
    n,lab,st,_=cv2.connectedComponentsWithStats(red*roi)
    tgt=np.zeros_like(red)
    for i in range(1,n):
        if st[i,4]>30: tgt|=(lab==i).astype(np.uint8)
    def loss(p):
        m=render(p,proj,red.shape,zt)*roi
        inter=float((m&tgt).sum()); uni=float((m|tgt).sum()); return -inter/max(uni,1.0)
    best=None
    for dx in(-0.01,0,0.01):
        for dy in(-0.01,0,0.01):
            for dr in(-10,0,10):
                p0=np.array(init)+[dx,dy,dr]
                r=minimize(loss,p0,method="Nelder-Mead",options=dict(xatol=1e-4,fatol=1e-4,initial_simplex=[p0,p0+[0.004,0,0],p0+[0,0.004,0],p0+[0,0,4]]))
                if best is None or r.fun<best.fun: best=r
    return best.x,-best.fun,tgt,roi
def scene_proj(P):
    p=(sc.inv(T_bs)@np.c_[P,np.ones(len(P))].T)[:3].T
    uv,_=cv2.projectPoints(p,np.zeros(3),np.zeros(3),Ks,Ds); return uv.reshape(-1,2)
def wrist_proj_fn(q):
    fk=sc.FK(); Tbc=fk.base_hand(np.array(q))@X
    def f(P):
        p=(sc.inv(Tbc)@np.c_[P,np.ones(len(P))].T)[:3].T
        if (p[:,2]<0.02).any(): return None
        uv,_=cv2.projectPoints(p,np.zeros(3),np.zeros(3),Kw,Dw); return uv.reshape(-1,2)
    return f

import numpy as np
pi=np.pi
def rot(axis,a):
    axis=np.asarray(axis,float); axis/=np.linalg.norm(axis); x,y,z=axis; c,s=np.cos(a),np.sin(a); C=1-c
    R=np.array([[c+x*x*C,x*y*C-z*s,x*z*C+y*s],[y*x*C+z*s,c+y*y*C,y*z*C-x*s],[z*x*C-y*s,z*y*C+x*s,c+z*z*C]])
    T=np.eye(4);T[:3,:3]=R;return T
def tr(x,y,z):
    T=np.eye(4);T[:3,3]=[x,y,z];return T
# --- URDF (all rpy = 0)
ORG=[(0,0,.33),(.04,0,0),(0,0,.445),(.44,0,.04),(0,0,0),(0,0,0)]
AX=[(0,0,1),(0,1,0),(0,-1,0),(-1,0,0),(0,-1,0),(-1,0,0)]
def urdf_frames(q):
    T=np.eye(4);out=[T.copy()]
    for o,a,qi in zip(ORG,AX,q):
        T=T@tr(*o)@rot(a,qi);out.append(T.copy())
    return out
# --- standard DH  (theta_offset, d, a, alpha)
DH=[(0,.33,.04,-pi/2),(-pi/2,0,.445,pi),(0,0,.04,-pi/2),(0,-.44,0,pi/2),(pi,0,0,pi/2),(0,-.08,0,pi)]
def A(th,d,a,al):
    return rot((0,0,1),th)@tr(0,0,d)@tr(a,0,0)@rot((1,0,0),al)
def dh_frames(q):
    T=np.eye(4);out=[T.copy()]
    for (off,d,a,al),qi in zip(DH,q):
        T=T@A(qi+off,d,a,al);out.append(T.copy())
    return out
if __name__=="__main__":
    rng=np.random.default_rng(1)
    lim=np.deg2rad([[-170,170],[-65,145],[-70,190],[-190,190],[-135,135],[-360,360]])
    worst_pos=0;worst_zaxis=0;worst_tool=0
    for _ in range(2000):
        q=rng.uniform(lim[:,0],lim[:,1])
        U=urdf_frames(q);D=dh_frames(q)
        # joint i+1 axis (world) from URDF vs z_i of DH frame i
        for i in range(6):
            ax=U[i][:3,:3]@ (np.array(AX[i]))      # axis of joint i+1 in world
            worst_zaxis=max(worst_zaxis,np.linalg.norm(ax-D[i][:3,2]))
        # origin of DH frame 4..6 coincides with wrist; DH frame 6 vs URDF flange
        flange=U[6]@tr(.08,0,0)
        worst_pos=max(worst_pos,np.linalg.norm(flange[:3,3]-D[6][:3,3]))
        # flange x-axis (tool dir) = +x of link6 frame = z6 of DH
        worst_tool=max(worst_tool,np.linalg.norm(flange[:3,0]-D[6][:3,2]))
    print("max flange position error [m]:",worst_pos)
    print("max joint-axis error (z_i vs URDF axis i+1):",worst_zaxis)
    print("max tool-direction error:",worst_tool)
    np.set_printoptions(precision=4,suppress=True)
    print("zero-pose flange (URDF):",(urdf_frames(np.zeros(6))[6]@tr(.08,0,0))[:3,3])

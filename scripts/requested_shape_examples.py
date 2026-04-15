
import numpy as np, pandas as pd, math, time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

def delta_cosine(s, eta):
    a=np.abs(s)
    out=np.zeros_like(s,dtype=float)
    m=a<=eta
    out[m]=0.5/eta*(1+np.cos(np.pi*s[m]/eta))
    return out

def segment_area_circle(R,h):
    if h<=0: return 0.0
    if h>=2*R: return np.pi*R*R
    return R*R*np.arccos((R-h)/R)-(R-h)*np.sqrt(max(0.0,2*R*h-h*h))

class SmallBoxPunch:
    name="small_cube"
    def __init__(self,lx,lz,H):
        self.lx,self.lz,self.H=lx,lz,H
        self.hx,self.hz,self.hy=lx/2,lz/2,H/2
    def exact_force(self,k,delta):
        return k*self.lx*self.lz*delta
    def bbox(self,delta):
        return (-self.hx,self.hx,-self.hz,self.hz)
    def phi(self,X,Y,Z,delta):
        cx,cy,cz=0.0,self.hy-delta,0.0
        qx=np.abs(X-cx)-self.hx
        qy=np.abs(Y-cy)-self.hy
        qz=np.abs(Z-cz)-self.hz
        ox=np.maximum(qx,0.0); oy=np.maximum(qy,0.0); oz=np.maximum(qz,0.0)
        outside=np.sqrt(ox*ox+oy*oy+oz*oz)
        inside=np.minimum(np.maximum.reduce([qx,qy,qz]),0.0)
        return outside+inside
    def direct_normal(self,X,Y,Z,delta,h):
        px=(self.phi(X+h,Y,Z,delta)-self.phi(X-h,Y,Z,delta))/(2*h)
        py=(self.phi(X,Y+h,Z,delta)-self.phi(X,Y-h,Z,delta))/(2*h)
        pz=(self.phi(X,Y,Z+h,delta)-self.phi(X,Y,Z-h,delta))/(2*h)
        gn=np.sqrt(px*px+py*py+pz*pz)+1e-15
        return px/gn,py/gn,pz/gn
    def local_quad(self,k,delta,cube_size,cube_height,Nxz=40,quad_order=2):
        x0,x1,z0,z1=self.bbox(delta)
        xs=np.linspace(x0,x1,Nxz+1); zs=np.linspace(z0,z1,Nxz+1)
        xi,wi=np.polynomial.legendre.leggauss(quad_order)
        Fy=0.0
        pts=[]
        for i in range(Nxz):
            xl,xr=xs[i],xs[i+1]
            for j in range(Nxz):
                zl,zr=zs[j],zs[j+1]
                for a,wa in zip(xi,wi):
                    xq=0.5*(xr-xl)*a+0.5*(xr+xl)
                    for b,wb in zip(xi,wi):
                        zq=0.5*(zr-zl)*b+0.5*(zr+zl)
                        wproj=0.25*(xr-xl)*(zr-zl)*wa*wb
                        Fy += k*delta * wproj
                        pts.append((xq,zq))
        return dict(Fx=0.0,Fy=Fy,Fz=0.0,Mx=0.0,My=0.0,Mz=0.0,accepted_pts=pts)

class AnnularPunch:
    name="annular_punch"
    def __init__(self,ri,ro,H):
        self.ri,self.ro,self.H=ri,ro,H
        self.hy=H/2
    def exact_force(self,k,delta):
        return k*np.pi*(self.ro*self.ro-self.ri*self.ri)*delta
    def bbox(self,delta):
        return (-self.ro,self.ro,-self.ro,self.ro)
    def sd_annulus_2d(self,X,Z):
        r=np.sqrt(X*X+Z*Z)
        out=np.empty_like(r,dtype=float)
        m1=r<self.ri; m2=(r>=self.ri)&(r<=self.ro); m3=r>self.ro
        out[m1]=self.ri-r[m1]
        out[m2]=-np.minimum(r[m2]-self.ri,self.ro-r[m2])
        out[m3]=r[m3]-self.ro
        return out
    def phi(self,X,Y,Z,delta):
        sd2=self.sd_annulus_2d(np.asarray(X),np.asarray(Z))
        yc=self.hy-delta
        qy=np.abs(np.asarray(Y)-yc)-self.hy
        ox=np.maximum(sd2,0.0); oy=np.maximum(qy,0.0)
        outside=np.sqrt(ox*ox+oy*oy)
        inside=np.minimum(np.maximum(sd2,qy),0.0)
        return outside+inside
    def direct_normal(self,X,Y,Z,delta,h):
        px=(self.phi(X+h,Y,Z,delta)-self.phi(X-h,Y,Z,delta))/(2*h)
        py=(self.phi(X,Y+h,Z,delta)-self.phi(X,Y-h,Z,delta))/(2*h)
        pz=(self.phi(X,Y,Z+h,delta)-self.phi(X,Y,Z-h,delta))/(2*h)
        gn=np.sqrt(px*px+py*py+pz*pz)+1e-15
        return px/gn,py/gn,pz/gn
    def local_quad(self,k,delta,cube_size,cube_height,Nxz=48,quad_order=3):
        x0,x1,z0,z1=self.bbox(delta)
        xs=np.linspace(x0,x1,Nxz+1); zs=np.linspace(z0,z1,Nxz+1)
        xi,wi=np.polynomial.legendre.leggauss(quad_order)
        Fy=0.0; pts=[]
        for i in range(Nxz):
            xl,xr=xs[i],xs[i+1]
            for j in range(Nxz):
                zl,zr=zs[j],zs[j+1]
                for a,wa in zip(xi,wi):
                    xq=0.5*(xr-xl)*a+0.5*(xr+xl)
                    for b,wb in zip(xi,wi):
                        zq=0.5*(zr-zl)*b+0.5*(zr+zl)
                        r=(xq*xq+zq*zq)**0.5
                        if not (self.ri<=r<=self.ro): 
                            continue
                        wproj=0.25*(xr-xl)*(zr-zl)*wa*wb
                        Fy += k*delta*wproj
                        pts.append((xq,zq))
        return dict(Fx=0.0,Fy=Fy,Fz=0.0,Mx=0.0,My=0.0,Mz=0.0,accepted_pts=pts)

class HorizontalCylinder:
    name="horizontal_cylinder"
    def __init__(self,R,L):
        self.R,self.L=R,L
    def exact_force(self,k,delta):
        return k*self.L*segment_area_circle(self.R,delta)
    def bbox(self,delta):
        a=np.sqrt(max(0.0,2*self.R*delta-delta*delta))
        return (-a,a,-self.L/2,self.L/2)
    def phi(self,X,Y,Z,delta):
        cx,cy,cz=0.0,self.R-delta,0.0
        d_rad=np.sqrt((X-cx)**2+(Y-cy)**2)-self.R
        d_ax=np.abs(Z-cz)-self.L/2
        ox=np.maximum(d_rad,0.0); oy=np.maximum(d_ax,0.0)
        outside=np.sqrt(ox*ox+oy*oy)
        inside=np.minimum(np.maximum(d_rad,d_ax),0.0)
        return outside+inside
    def direct_normal(self,X,Y,Z,delta,h):
        cx,cy=0.0,self.R-delta
        rr=np.sqrt((X-cx)**2+(Y-cy)**2)+1e-15
        return (X-cx)/rr,(Y-cy)/rr,np.zeros_like(Z)
    def local_quad(self,k,delta,cube_size,cube_height,Nxz=48,quad_order=3):
        x0,x1,z0,z1=self.bbox(delta)
        xs=np.linspace(x0,x1,Nxz+1); zs=np.linspace(z0,z1,Nxz+1)
        xi,wi=np.polynomial.legendre.leggauss(quad_order)
        cy=self.R-delta
        Fy=0.0; pts=[]
        for i in range(Nxz):
            xl,xr=xs[i],xs[i+1]
            for j in range(Nxz):
                zl,zr=zs[j],zs[j+1]
                for a,wa in zip(xi,wi):
                    xq=0.5*(xr-xl)*a+0.5*(xr+xl)
                    for b,wb in zip(xi,wi):
                        zq=0.5*(zr-zl)*b+0.5*(zr+zl)
                        if abs(zq)>self.L/2 or abs(xq)>np.sqrt(max(0.0,2*self.R*delta-delta*delta))+1e-12:
                            continue
                        y_sigma=cy-np.sqrt(max(0.0,self.R*self.R-xq*xq))
                        ny=(y_sigma-cy)/self.R
                        wproj=0.25*(xr-xl)*(zr-zl)*wa*wb
                        dA=wproj/abs(ny)
                        p=k*max(0.0,-y_sigma)
                        Fy += -p*ny*dA
                        pts.append((xq,zq))
        return dict(Fx=0.0,Fy=Fy,Fz=0.0,Mx=0.0,My=0.0,Mz=0.0,accepted_pts=pts)

def segment_area_circle(R,h):
    if h<=0: return 0.0
    if h>=2*R: return np.pi*R*R
    return R*R*np.arccos((R-h)/R)-(R-h)*np.sqrt(max(0.0,2*R*h-h*h))

def direct_band_integral(shape,delta,k,cube_size,cube_height,N=64,eta_factor=1.5):
    x_min,x_max=-cube_size/2,cube_size/2
    z_min,z_max=-cube_size/2,cube_size/2
    y_min,y_max=-cube_height,0.0
    xs=np.linspace(x_min,x_max,N,endpoint=False)+(x_max-x_min)/N/2
    ys=np.linspace(y_min,y_max,N,endpoint=False)+(y_max-y_min)/N/2
    zs=np.linspace(z_min,z_max,N,endpoint=False)+(z_max-z_min)/N/2
    dx=(x_max-x_min)/N; dy=(y_max-y_min)/N; dz=(z_max-z_min)/N; dV=dx*dy*dz
    X,Y,Z=np.meshgrid(xs,ys,zs,indexing='ij')
    phi=shape.phi(X,Y,Z,delta)
    h=max(dx,dy,dz); eta=eta_factor*h
    band=delta_cosine(phi,eta)
    nx,ny,nz=shape.direct_normal(X,Y,Z,delta,h)
    p=k*np.clip(-Y,0.0,None)
    tx=-p*nx; ty=-p*ny; tz=-p*nz
    Fx=np.sum(tx*band)*dV
    Fy=np.sum(ty*band)*dV
    Fz=np.sum(tz*band)*dV
    Mx=np.sum((Y*tz-Z*ty)*band)*dV
    My=np.sum((Z*tx-X*tz)*band)*dV
    Mz=np.sum((X*ty-Y*tx)*band)*dV
    return dict(Fx=Fx,Fy=Fy,Fz=Fz,Mx=Mx,My=My,Mz=Mz)

def run_all(out_dir):
    cube_size=1.8; cube_height=0.25; k=10.0
    deltas=np.array([0.01,0.02,0.03,0.05,0.08])
    shapes=[
        ("small_cube",SmallBoxPunch(0.45,0.45,0.40)),
        ("annular_punch",AnnularPunch(0.12,0.30,0.40)),
        ("horizontal_cylinder",HorizontalCylinder(0.28,0.80)),
    ]
    rows=[]
    # sweeps
    for name,shape in shapes:
        for delta in deltas:
            Fy_exact=shape.exact_force(k,delta)
            t0=time.perf_counter(); A=direct_band_integral(shape,delta,k,cube_size,cube_height,N=64); t1=time.perf_counter()
            t2=time.perf_counter(); B=shape.local_quad(k,delta,cube_size,cube_height,Nxz=40 if name=="small_cube" else 48,quad_order=3 if name!="small_cube" else 2); t3=time.perf_counter()
            rows.append(dict(case_family=name+"_sweep",method="A_direct_band",delta=delta,resolution=64,Fy_num=A['Fy'],Fy_exact=Fy_exact,rel_err_Fy=abs(A['Fy']-Fy_exact)/abs(Fy_exact),time_sec=t1-t0))
            rows.append(dict(case_family=name+"_sweep",method="B_local_normal_quad",delta=delta,resolution=40 if name=="small_cube" else 48,Fy_num=B['Fy'],Fy_exact=Fy_exact,rel_err_Fy=abs(B['Fy']-Fy_exact)/abs(Fy_exact),time_sec=t3-t2))
        delta_conv=0.03; Fy_exact=shape.exact_force(k,delta_conv)
        for N in [40,56,72,88]:
            t0=time.perf_counter(); A=direct_band_integral(shape,delta_conv,k,cube_size,cube_height,N=N); t1=time.perf_counter()
            rows.append(dict(case_family=name+"_conv",method="A_direct_band",delta=delta_conv,resolution=N,Fy_num=A['Fy'],Fy_exact=Fy_exact,rel_err_Fy=abs(A['Fy']-Fy_exact)/abs(Fy_exact),time_sec=t1-t0))
        for Nxz in [20,28,40,56]:
            t0=time.perf_counter(); B=shape.local_quad(k,delta_conv,cube_size,cube_height,Nxz=Nxz,quad_order=3 if name!="small_cube" else 2); t1=time.perf_counter()
            rows.append(dict(case_family=name+"_conv",method="B_local_normal_quad",delta=delta_conv,resolution=Nxz,Fy_num=B['Fy'],Fy_exact=Fy_exact,rel_err_Fy=abs(B['Fy']-Fy_exact)/abs(Fy_exact),time_sec=t1-t0))
    df=pd.DataFrame(rows)
    df.to_csv(out_dir/'requested_shape_examples_results.csv',index=False)

    # geometry demo
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.0))
    for ax,(name,shape) in zip(axes,shapes):
        B=shape.local_quad(k,0.03,cube_size,cube_height,Nxz=36,quad_order=3 if name!="small_cube" else 2)
        pts=np.array(B['accepted_pts']) if B['accepted_pts'] else np.zeros((0,2))
        if len(pts)>0: ax.scatter(pts[:,0],pts[:,1],s=6)
        ax.set_aspect('equal'); ax.set_title(name); ax.set_xlabel('x'); ax.set_ylabel('z')
    plt.tight_layout(); plt.savefig(out_dir/'requested_shape_examples_geometry_demo.png',dpi=180); plt.close()

    for name,_ in shapes:
        sub=df[df['case_family']==name+'_sweep'].copy()
        plt.figure(figsize=(7.0,4.5))
        for method,grp in sub.groupby('method'):
            grp=grp.sort_values('delta')
            plt.plot(grp['delta'],grp['Fy_num'],marker='o',label=method)
        exact=sub.drop_duplicates('delta').sort_values('delta')
        plt.plot(exact['delta'],exact['Fy_exact'],marker='x',label='exact')
        plt.xlabel('penetration depth δ'); plt.ylabel('vertical force Fy'); plt.title(name.replace('_',' ') + ': shallow force comparison'); plt.legend(); plt.tight_layout()
        fname={'small_cube':'requested_small_cube_force_compare.png','annular_punch':'requested_annular_punch_force_compare.png','horizontal_cylinder':'requested_horizontal_cylinder_force_compare.png'}[name]
        plt.savefig(out_dir/fname,dpi=180); plt.close()

    fig,axes=plt.subplots(1,3,figsize=(12.8,4.1))
    for ax,(name,_) in zip(axes,shapes):
        sub=df[df['case_family']==name+'_conv'].copy()
        for method,grp in sub.groupby('method'):
            grp=grp.sort_values('resolution')
            ax.plot(grp['resolution'],grp['rel_err_Fy'],marker='o',label=method)
        ax.set_yscale('log'); ax.set_title(name.replace('_',' ')); ax.set_xlabel('resolution'); ax.set_ylabel('relative Fy error'); ax.legend()
    plt.tight_layout(); plt.savefig(out_dir/'requested_shape_examples_convergence.png',dpi=180); plt.close()

    return df

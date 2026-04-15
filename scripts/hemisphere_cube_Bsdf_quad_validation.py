import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import time

def exact_cap_volume(R,d): return np.pi*d*d*(R-d/3.0)
def analytic_force(R,d,k): return k*exact_cap_volume(R,d)
def sphere_phi_xyz(x,y,z,R,delta):
    cy=R-delta
    return np.sqrt(x*x+(y-cy)*(y-cy)+z*z)-R

def sphere_normal_xyz(x,y,z,R,delta):
    cy=R-delta
    dx,dy,dz=x,y-cy,z
    rr=np.sqrt(dx*dx+dy*dy+dz*dz)
    return np.array([dx,dy,dz])/max(rr,1e-15)

def delta_cosine(s,eta):
    s=np.asarray(s); a=np.abs(s); out=np.zeros_like(s,dtype=float)
    m=a<=eta
    out[m]=0.5/eta*(1+np.cos(np.pi*s[m]/eta))
    return out

def version_A_direct_band(R,delta,k,cube_size,cube_height,N,eta_factor=1.5):
    x0,x1=-cube_size/2,cube_size/2; z0,z1=x0,x1; y0,y1=-cube_height,0.0
    xs=np.linspace(x0,x1,N,endpoint=False)+(x1-x0)/N/2
    ys=np.linspace(y0,y1,N,endpoint=False)+(y1-y0)/N/2
    zs=np.linspace(z0,z1,N,endpoint=False)+(z1-z0)/N/2
    dx=(x1-x0)/N; dy=(y1-y0)/N; dz=(z1-z0)/N; dV=dx*dy*dz
    X,Y,Z=np.meshgrid(xs,ys,zs,indexing='ij')
    cy=R-delta
    RR=np.sqrt(X*X+(Y-cy)*(Y-cy)+Z*Z)
    phi=RR-R
    nx=X/np.maximum(RR,1e-15); ny=(Y-cy)/np.maximum(RR,1e-15); nz=Z/np.maximum(RR,1e-15)
    p=k*np.clip(-Y,0,None)
    eta=eta_factor*max(dx,dy,dz)
    band=delta_cosine(phi,eta)
    tx,ty,tz=-p*nx,-p*ny,-p*nz
    Fx=np.sum(tx*band)*dV; Fy=np.sum(ty*band)*dV; Fz=np.sum(tz*band)*dV
    Mx=np.sum((Y*tz-Z*ty)*band)*dV; My=np.sum((Z*tx-X*tz)*band)*dV; Mz=np.sum((X*ty-Y*tx)*band)*dV
    return dict(Fx=Fx,Fy=Fy,Fz=Fz,Mx=Mx,My=My,Mz=Mz)

def version_B_analytic_sheet(R,delta,k,Na,Nt):
    amax=np.arccos(1-delta/R)
    if amax<=0: return dict(Fx=0.,Fy=0.,Fz=0.,Mx=0.,My=0.,Mz=0.)
    da=amax/Na; dt=2*np.pi/Nt
    a=(np.arange(Na)+0.5)*da; t=(np.arange(Nt)+0.5)*dt
    A,T=np.meshgrid(a,t,indexing='ij')
    cy=R-delta
    X=R*np.sin(A)*np.cos(T); Y=cy-R*np.cos(A); Z=R*np.sin(A)*np.sin(T)
    nx=np.sin(A)*np.cos(T); ny=-np.cos(A); nz=np.sin(A)*np.sin(T)
    p=k*np.clip(-Y,0,None); dA=R*R*np.sin(A)*da*dt
    dFx,dFy,dFz=-p*nx*dA,-p*ny*dA,-p*nz*dA
    Fx,Fy,Fz=np.sum(dFx),np.sum(dFy),np.sum(dFz)
    Mx=np.sum(Y*dFz-Z*dFy); My=np.sum(Z*dFx-X*dFz); Mz=np.sum(X*dFy-Y*dFx)
    return dict(Fx=Fx,Fy=Fy,Fz=Fz,Mx=Mx,My=My,Mz=Mz)

def poly_signed_area(poly):
    if len(poly)<3: return 0.0
    x,y=poly[:,0],poly[:,1]; xp=np.roll(x,-1); yp=np.roll(y,-1)
    return 0.5*np.sum(x*yp-xp*y)

def poly_area_centroid(poly):
    if len(poly)<3: return 0.0,np.array([np.nan,np.nan])
    x,y=poly[:,0],poly[:,1]; xp=np.roll(x,-1); yp=np.roll(y,-1)
    cross=x*yp-xp*y; A=0.5*np.sum(cross)
    if abs(A)<1e-15: return 0.0,np.array([np.nan,np.nan])
    Cx=np.sum((x+xp)*cross)/(6*A); Cy=np.sum((y+yp)*cross)/(6*A)
    return abs(A),np.array([Cx,Cy])

def cross2(a,b): return a[0]*b[1]-a[1]*b[0]
def triangle_area(tri): return 0.5*abs(cross2(tri[1]-tri[0],tri[2]-tri[0]))

def line_intersection(p1,p2,a,b):
    r=p2-p1; s=b-a; denom=cross2(r,s)
    if abs(denom)<1e-14: return p2.copy()
    t=cross2(a-p1,s)/denom
    return p1+t*r

def clip_against_edge(poly,a,b):
    if len(poly)==0: return poly
    out=[]; prev=poly[-1]; prev_in=cross2(b-a,prev-a)>=-1e-12
    for curr in poly:
        curr_in=cross2(b-a,curr-a)>=-1e-12
        if curr_in:
            if not prev_in: out.append(line_intersection(prev,curr,a,b))
            out.append(curr)
        elif prev_in:
            out.append(line_intersection(prev,curr,a,b))
        prev=curr; prev_in=curr_in
    return np.array(out,dtype=float)

def clip_square_by_convex_polygon(square,poly):
    p=square.copy()
    for i in range(len(poly)):
        p=clip_against_edge(p,poly[i],poly[(i+1)%len(poly)])
        if len(p)==0: break
    return p

def point_in_convex(pt,poly):
    for i in range(len(poly)):
        if cross2(poly[(i+1)%len(poly)]-poly[i], pt-poly[i]) < -1e-12: return False
    return True

def resample_closed_polygon(poly,M):
    if len(poly)<3: return poly
    closed=np.vstack([poly,poly[0]])
    seg=np.linalg.norm(np.diff(closed,axis=0),axis=1)
    s=np.concatenate([[0.0],np.cumsum(seg)]); total=s[-1]
    if total<=0: return poly
    targets=np.linspace(0,total,M+1)[:-1]
    out=[]; j=0
    for t in targets:
        while j+1<len(s) and s[j+1]<t: j+=1
        a,b=closed[j],closed[j+1]; ds=s[j+1]-s[j]; w=0 if ds<=0 else (t-s[j])/ds
        out.append((1-w)*a+w*b)
    out=np.array(out)
    if poly_signed_area(out)<0: out=out[::-1]
    return out

def triangulate_convex_polygon(poly):
    if len(poly)<3: return []
    _,c=poly_area_centroid(poly); tris=[]
    for i in range(len(poly)):
        tri=np.vstack([c,poly[i],poly[(i+1)%len(poly)]])
        if triangle_area(tri)>1e-15: tris.append(tri)
    return tris

def triangle_quadrature_points_3pt(tri):
    A=triangle_area(tri); v0,v1,v2=tri
    bary=np.array([[2/3,1/6,1/6],[1/6,2/3,1/6],[1/6,1/6,2/3]],float)
    w=A/3.0
    return [(lam[0]*v0+lam[1]*v1+lam[2]*v2,w) for lam in bary]

def extract_top_slice_contour_polygon(R,delta,cube_size,Ncontour=240,Mpoly=64):
    x0,x1=-cube_size/2,cube_size/2; z0,z1=x0,x1
    xs=np.linspace(x0,x1,Ncontour); zs=np.linspace(z0,z1,Ncontour)
    X,Z=np.meshgrid(xs,zs,indexing='xy')
    phi=sphere_phi_xyz(X,0.0,Z,R,delta)
    fig,ax=plt.subplots()
    cs=ax.contour(xs,zs,phi,levels=[0.0])
    segs=cs.allsegs[0]
    plt.close(fig)
    if not segs: return np.zeros((0,2))
    poly=max(segs,key=lambda a:a.shape[0]).copy()
    if np.linalg.norm(poly[0]-poly[-1])>1e-12: poly=np.vstack([poly,poly[0]])
    poly=poly[:-1]
    if poly_signed_area(poly)<0: poly=poly[::-1]
    return resample_closed_polygon(poly,Mpoly)

def solve_root_y(x,z,R,delta,cube_height,steps=28):
    phi_top=sphere_phi_xyz(x,0.0,z,R,delta); phi_bot=sphere_phi_xyz(x,-cube_height,z,R,delta)
    if not (phi_top<=0 and phi_bot>=0): return None
    yl,yr=-cube_height,0.0
    for _ in range(steps):
        ym=0.5*(yl+yr); fm=sphere_phi_xyz(x,ym,z,R,delta)
        if fm>0: yl=ym
        else: yr=ym
    return 0.5*(yl+yr)

def version_B_sdf_column_center(R,delta,k,cube_size,cube_height,Nxz):
    x0,x1=-cube_size/2,cube_size/2; z0,z1=x0,x1
    xs=np.linspace(x0,x1,Nxz,endpoint=False)+(x1-x0)/Nxz/2
    zs=np.linspace(z0,z1,Nxz,endpoint=False)+(z1-z0)/Nxz/2
    dx=(x1-x0)/Nxz; dz=(z1-z0)/Nxz
    Fx=Fy=Fz=Mx=My=Mz=0.0
    for x in xs:
        for z in zs:
            y=solve_root_y(x,z,R,delta,cube_height)
            if y is None: continue
            n=sphere_normal_xyz(x,y,z,R,delta); dA=dx*dz/abs(n[1]); p=k*max(0,-y)
            dF=-p*n*dA; Fx+=dF[0]; Fy+=dF[1]; Fz+=dF[2]
            M=np.cross(np.array([x,y,z]),dF); Mx+=M[0]; My+=M[1]; Mz+=M[2]
    return dict(Fx=Fx,Fy=Fy,Fz=Fz,Mx=Mx,My=My,Mz=Mz)

def version_B_sdf_clipped_centroid(R,delta,k,cube_size,cube_height,Nxz,Ncontour=240,Mpoly=64):
    poly=extract_top_slice_contour_polygon(R,delta,cube_size,Ncontour,Mpoly)
    if len(poly)==0: return dict(Fx=0.,Fy=0.,Fz=0.,Mx=0.,My=0.,Mz=0.,poly=poly,pts=[])
    x0,x1=-cube_size/2,cube_size/2; z0,z1=x0,x1
    x_edges=np.linspace(x0,x1,Nxz+1); z_edges=np.linspace(z0,z1,Nxz+1)
    dx=x_edges[1]-x_edges[0]; dz=z_edges[1]-z_edges[0]; cell_area=dx*dz
    ix0=max(0,int(np.floor((poly[:,0].min()-x0)/dx))-1); ix1=min(Nxz-1,int(np.floor((poly[:,0].max()-x0)/dx))+1)
    iz0=max(0,int(np.floor((poly[:,1].min()-z0)/dz))-1); iz1=min(Nxz-1,int(np.floor((poly[:,1].max()-z0)/dz))+1)
    Fx=Fy=Fz=Mx=My=Mz=0.0; pts=[]
    for i in range(ix0,ix1+1):
        for j in range(iz0,iz1+1):
            square=np.array([[x_edges[i],z_edges[j]],[x_edges[i+1],z_edges[j]],[x_edges[i+1],z_edges[j+1]],[x_edges[i],z_edges[j+1]]],float)
            ins=[point_in_convex(c,poly) for c in square]
            if all(ins):
                Aproj=cell_area; C=np.array([0.5*(x_edges[i]+x_edges[i+1]),0.5*(z_edges[j]+z_edges[j+1])])
            else:
                clipped=clip_square_by_convex_polygon(square,poly)
                Aproj,C=poly_area_centroid(clipped)
                if Aproj<=1e-15: continue
            x,z=float(C[0]),float(C[1]); y=solve_root_y(x,z,R,delta,cube_height)
            if y is None: continue
            n=sphere_normal_xyz(x,y,z,R,delta); dA=Aproj/abs(n[1]); p=k*max(0,-y)
            dF=-p*n*dA; Fx+=dF[0]; Fy+=dF[1]; Fz+=dF[2]
            M=np.cross(np.array([x,y,z]),dF); Mx+=M[0]; My+=M[1]; Mz+=M[2]
            pts.append((x,y,z))
    return dict(Fx=Fx,Fy=Fy,Fz=Fz,Mx=Mx,My=My,Mz=Mz,poly=poly,pts=pts)

def version_B_sdf_quad_sheet(R,delta,k,cube_size,cube_height,Nxz,Ncontour=240,Mpoly=64):
    poly=extract_top_slice_contour_polygon(R,delta,cube_size,Ncontour,Mpoly)
    if len(poly)==0: return dict(Fx=0.,Fy=0.,Fz=0.,Mx=0.,My=0.,Mz=0.,poly=poly,qps=[])
    x0,x1=-cube_size/2,cube_size/2; z0,z1=x0,x1
    x_edges=np.linspace(x0,x1,Nxz+1); z_edges=np.linspace(z0,z1,Nxz+1)
    dx=x_edges[1]-x_edges[0]; dz=z_edges[1]-z_edges[0]; cell_area=dx*dz
    ix0=max(0,int(np.floor((poly[:,0].min()-x0)/dx))-1); ix1=min(Nxz-1,int(np.floor((poly[:,0].max()-x0)/dx))+1)
    iz0=max(0,int(np.floor((poly[:,1].min()-z0)/dz))-1); iz1=min(Nxz-1,int(np.floor((poly[:,1].max()-z0)/dz))+1)
    Fx=Fy=Fz=Mx=My=Mz=0.0; qps=[]
    for i in range(ix0,ix1+1):
        for j in range(iz0,iz1+1):
            square=np.array([[x_edges[i],z_edges[j]],[x_edges[i+1],z_edges[j]],[x_edges[i+1],z_edges[j+1]],[x_edges[i],z_edges[j+1]]],float)
            ins=[point_in_convex(c,poly) for c in square]
            if all(ins): clipped=square
            else:
                clipped=clip_square_by_convex_polygon(square,poly)
                if len(clipped)<3: continue
            for tri in triangulate_convex_polygon(clipped):
                for p2d,w_proj in triangle_quadrature_points_3pt(tri):
                    x,z=float(p2d[0]),float(p2d[1]); y=solve_root_y(x,z,R,delta,cube_height)
                    if y is None: continue
                    n=sphere_normal_xyz(x,y,z,R,delta); dA=w_proj/abs(n[1]); p=k*max(0,-y)
                    dF=-p*n*dA; Fx+=dF[0]; Fy+=dF[1]; Fz+=dF[2]
                    M=np.cross(np.array([x,y,z]),dF); Mx+=M[0]; My+=M[1]; Mz+=M[2]
                    qps.append((x,y,z))
    return dict(Fx=Fx,Fy=Fy,Fz=Fz,Mx=Mx,My=My,Mz=Mz,poly=poly,qps=qps)

def make_geometry_demo(path,poly,qps,cube_size):
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    fig=plt.figure(figsize=(12,5))
    ax1=fig.add_subplot(1,2,1)
    if len(poly)>0:
        pc=np.vstack([poly,poly[0]]); ax1.plot(pc[:,0],pc[:,1],linewidth=2,label='footprint contour')
    if qps:
        arr=np.array(qps); ax1.scatter(arr[:,0],arr[:,2],s=8,label='quadrature pts')
    ax1.set_aspect('equal'); ax1.set_xlim(-cube_size/2,cube_size/2); ax1.set_ylim(-cube_size/2,cube_size/2)
    ax1.set_xlabel('x'); ax1.set_ylabel('z'); ax1.set_title('Footprint clipping + 2D quadrature'); ax1.legend()
    ax2=fig.add_subplot(1,2,2,projection='3d')
    if qps:
        arr=np.array(qps); ax2.scatter(arr[:,0],arr[:,1],arr[:,2],s=8)
    ax2.set_xlabel('x'); ax2.set_ylabel('y'); ax2.set_zlabel('z'); ax2.set_title('Recovered local-normal quadrature points'); ax2.set_box_aspect((1,0.6,1))
    plt.tight_layout(); plt.savefig(path,dpi=180); plt.close()

def run_all(out_dir):
    R=1.0; k=10.0; cube_size=1.6; cube_height=0.25
    shallow_deltas=np.array([0.01,0.02,0.03,0.05,0.08,0.12])
    rows=[]; geom=None
    for delta in shallow_deltas:
        Fy_exact=analytic_force(R,delta,k)
        configs=[
            ('A_direct_band', lambda: version_A_direct_band(R,delta,k,cube_size,cube_height,96), 96),
            ('B_analytic_sheet', lambda: version_B_analytic_sheet(R,delta,k,24,96), 24),
            ('B_sdf_column_center', lambda: version_B_sdf_column_center(R,delta,k,cube_size,cube_height,64), 64),
            ('B_sdf_clipped_centroid', lambda: version_B_sdf_clipped_centroid(R,delta,k,cube_size,cube_height,48), 48),
            ('B_sdf_quad_sheet', lambda: version_B_sdf_quad_sheet(R,delta,k,cube_size,cube_height,48), 48),
        ]
        for name,fn,res in configs:
            t0=time.perf_counter(); out=fn(); t1=time.perf_counter()
            rows.append(dict(mode='shallow_sweep',method=name,delta=delta,resolution=res,Fy_num=out['Fy'],Fy_exact=Fy_exact,
                             rel_err_Fy=abs(out['Fy']-Fy_exact)/abs(Fy_exact),
                             sym_resid=np.sqrt(out['Fx']**2+out['Fz']**2+out['Mx']**2+out['Mz']**2),
                             time_sec=t1-t0))
            if name=='B_sdf_quad_sheet' and abs(delta-0.03)<1e-12: geom=(out['poly'],out['qps'])
    delta_conv=0.03; Fy_exact=analytic_force(R,delta_conv,k)
    for N in [48,64,80,96,120,144]:
        t0=time.perf_counter(); out=version_A_direct_band(R,delta_conv,k,cube_size,cube_height,N); t1=time.perf_counter()
        rows.append(dict(mode='convergence',method='A_direct_band',delta=delta_conv,resolution=N,Fy_num=out['Fy'],Fy_exact=Fy_exact,
                         rel_err_Fy=abs(out['Fy']-Fy_exact)/abs(Fy_exact),sym_resid=np.sqrt(out['Fx']**2+out['Fz']**2+out['Mx']**2+out['Mz']**2),time_sec=t1-t0))
    for Na in [6,8,10,12,16,24,32,48]:
        t0=time.perf_counter(); out=version_B_analytic_sheet(R,delta_conv,k,Na,4*Na); t1=time.perf_counter()
        rows.append(dict(mode='convergence',method='B_analytic_sheet',delta=delta_conv,resolution=Na,Fy_num=out['Fy'],Fy_exact=Fy_exact,
                         rel_err_Fy=abs(out['Fy']-Fy_exact)/abs(Fy_exact),sym_resid=np.sqrt(out['Fx']**2+out['Fz']**2+out['Mx']**2+out['Mz']**2),time_sec=t1-t0))
    for Nxz in [12,16,24,32,48,64,96]:
        t0=time.perf_counter(); out=version_B_sdf_column_center(R,delta_conv,k,cube_size,cube_height,Nxz); t1=time.perf_counter()
        rows.append(dict(mode='convergence',method='B_sdf_column_center',delta=delta_conv,resolution=Nxz,Fy_num=out['Fy'],Fy_exact=Fy_exact,
                         rel_err_Fy=abs(out['Fy']-Fy_exact)/abs(Fy_exact),sym_resid=np.sqrt(out['Fx']**2+out['Fz']**2+out['Mx']**2+out['Mz']**2),time_sec=t1-t0))
    for Nxz in [12,16,24,32,48,64]:
        t0=time.perf_counter(); out=version_B_sdf_clipped_centroid(R,delta_conv,k,cube_size,cube_height,Nxz); t1=time.perf_counter()
        rows.append(dict(mode='convergence',method='B_sdf_clipped_centroid',delta=delta_conv,resolution=Nxz,Fy_num=out['Fy'],Fy_exact=Fy_exact,
                         rel_err_Fy=abs(out['Fy']-Fy_exact)/abs(Fy_exact),sym_resid=np.sqrt(out['Fx']**2+out['Fz']**2+out['Mx']**2+out['Mz']**2),time_sec=t1-t0))
    for Nxz in [12,16,24,32,48,64]:
        t0=time.perf_counter(); out=version_B_sdf_quad_sheet(R,delta_conv,k,cube_size,cube_height,Nxz); t1=time.perf_counter()
        rows.append(dict(mode='convergence',method='B_sdf_quad_sheet',delta=delta_conv,resolution=Nxz,Fy_num=out['Fy'],Fy_exact=Fy_exact,
                         rel_err_Fy=abs(out['Fy']-Fy_exact)/abs(Fy_exact),sym_resid=np.sqrt(out['Fx']**2+out['Fz']**2+out['Mx']**2+out['Mz']**2),time_sec=t1-t0))
    df=pd.DataFrame(rows)
    df.to_csv(out_dir/'hemisphere_cube_Bsdf_quad_results.csv',index=False)
    sweep=df[df['mode']=='shallow_sweep'].copy()
    plt.figure(figsize=(7.3,4.8))
    for method,grp in sweep.groupby('method'):
        grp=grp.sort_values('delta'); plt.plot(grp['delta'],grp['Fy_num'],marker='o',label=method)
    exact_curve=sweep.drop_duplicates('delta').sort_values('delta')
    plt.plot(exact_curve['delta'],exact_curve['Fy_exact'],marker='x',label='exact')
    plt.xlabel('penetration depth δ'); plt.ylabel('vertical force Fy'); plt.title('Shallow indentation: quadrature upgrade'); plt.legend(); plt.tight_layout(); plt.savefig(out_dir/'hemisphere_cube_Bsdf_quad_shallow_force_compare.png',dpi=180); plt.close()
    conv=df[df['mode']=='convergence'].copy()
    plt.figure(figsize=(7.3,4.8))
    for method,grp in conv.groupby('method'):
        grp=grp.sort_values('resolution'); plt.plot(grp['resolution'],grp['rel_err_Fy'],marker='o',label=method)
    plt.yscale('log'); plt.xlabel('resolution parameter'); plt.ylabel('relative Fy error'); plt.title('Shallow case convergence at δ=0.03'); plt.legend(); plt.tight_layout(); plt.savefig(out_dir/'hemisphere_cube_Bsdf_quad_convergence.png',dpi=180); plt.close()
    plt.figure(figsize=(7.3,4.8))
    for method,grp in conv.groupby('method'):
        grp=grp.sort_values('time_sec'); plt.plot(grp['time_sec'],grp['rel_err_Fy'],marker='o',label=method)
    plt.xscale('log'); plt.yscale('log'); plt.xlabel('runtime [s]'); plt.ylabel('relative Fy error'); plt.title('Accuracy vs runtime at shallow indentation'); plt.legend(); plt.tight_layout(); plt.savefig(out_dir/'hemisphere_cube_Bsdf_quad_error_vs_runtime.png',dpi=180); plt.close()
    plt.figure(figsize=(7.3,4.8))
    for method,grp in conv.groupby('method'):
        grp=grp.sort_values('resolution'); plt.plot(grp['resolution'],grp['sym_resid'],marker='o',label=method)
    plt.yscale('log'); plt.xlabel('resolution parameter'); plt.ylabel('symmetry residual'); plt.title('Centered axisymmetric case residuals'); plt.legend(); plt.tight_layout(); plt.savefig(out_dir/'hemisphere_cube_Bsdf_quad_symmetry_residual.png',dpi=180); plt.close()
    if geom is not None: make_geometry_demo(out_dir/'hemisphere_cube_Bsdf_quad_geometry_demo.png',geom[0],geom[1],cube_size)

if __name__=='__main__':
    run_all(Path(__file__).resolve().parent)

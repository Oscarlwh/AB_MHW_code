import numpy as np
from scipy.interpolate import CubicSpline



def Laplac_SOR(forcing,sigma,SC,CRI,dx,dy,p,lat,forcing_type):
    
    #Earth's radius, units: m
    Ra = 6.371e+06
    #Gas constant of dry air, unit: J/(kg.K)
    R_constant = 287
    
    nx,ny,np0=forcing.shape[0], forcing.shape[1], forcing.shape[2] #dims for Input forcing
    GT = np.zeros([nx,ny,np0])
    
    #Vertical interpolation
    dp=-2500.0
    pp=np.arange(100000.0, 9900, dp)
    nz=len(pp)
    
    #Radian conversion
    lat = lat*np.pi/180
    new_lat = np.tile(lat.reshape((1,ny,1)),(nx,1,nz))
    
    #Coriolis para.
    f = 2*(2*np.pi/24/60/60)*np.sin(lat);
    new_f=np.tile(f.reshape((1,ny,1)),(nx,1,nz))
    
    #Vertical interpolation for sigma
    new_R1=np.zeros([nx,ny,nz])
    new_pp=np.zeros([nx,ny,nz])
    
    
    
    cs = CubicSpline(np.flip(p),np.flip(sigma))
    R1=np.flip(cs(np.flip(pp)))
    R=(1/R1).reshape(1,1,nz)
    for iz in range(nz):
        new_R1[:,:,iz]=R1[iz]
        new_pp[:,:,iz]=pp[iz]
    
   #Vertical interpolation for forcing terms
    FRC0=np.zeros([nx,ny,nz])
    for ix in range(nx):
        for iy in range(ny):
            cs = CubicSpline(np.flip(p),np.flip(forcing[ix,iy,:]))
            FRC0[ix,iy,:]=np.flip(cs(np.flip(pp)))
            
            
    #Forcing type judgment
    if forcing_type==1:
        s1=FRC0/(new_pp*new_R1)
        f_dp=np.gradient(s1, 1, axis=2)
        s2=f_dp/dp
        FRC1=-new_f*s2*R_constant
        
    if forcing_type==2:
        FRC1=FRC0  
        
    
    FRC=(Ra*Ra)*(FRC1*new_f*np.cos(new_lat)*np.cos(new_lat));

    GT0=np.zeros([nx,ny,nz])
    GT1=np.zeros([nx,ny,nz])
    
    
    # #SOR Iteration
    # for k1 in range(500):
    #     GT1=GT0
    #     for iz in range(1, nz-1):
    #         for iy in range(1, ny-1):
    #             for ix in range(1, nx-1):
    #                 COE011=1/(dx*dx)
    #                 COE211=1/(dx*dx)
    #                 COE101=np.cos(new_lat[ix,iy,iz])*np.sin(new_lat[ix,iy,iz])/(2*dy)+np.cos(new_lat[ix,iy,iz])*np.cos(new_lat[ix,iy,iz])/(dy*dy)
    #                 COE121=-np.cos(new_lat[ix,iy,iz])*np.sin(new_lat[ix,iy,iz])/(2*dy)+np.cos(new_lat[ix,iy,iz])*np.cos(new_lat[ix,iy,iz])/(dy*dy)
    #                 COE110=-Ra*Ra*(np.cos(new_lat[ix,iy,iz])*np.cos(new_lat[ix,iy,iz])*new_f[ix,iy,iz]*new_f[ix,iy,iz])*(R[iz+1]-R[iz-1])/(2*dp)/(2*dp)+Ra*Ra*np.cos(new_lat[ix,iy,iz])*np.cos(new_lat[ix,iy,iz])*new_f[ix,iy,iz]*new_f[iz,iy,iz]*R[iz]/(dp*dp)
    #                 COE112=Ra*Ra*(np.cos(new_lat[ix,iy,iz])*np.cos(new_lat[ix,iy,iz])*new_f[ix,iy,iz]*new_f[ix,iy,iz])*(R[iz+1]-R[iz-1])/(2*dp)/(2*dp)+Ra*Ra*np.cos(new_lat[ix,iy,iz])*np.cos(new_lat[ix,iy,iz])*new_f[ix,iy,iz]*new_f[iz,iy,iz]*R[iz]/(dp*dp)
    #                 COE111=-2/(dx*dx)-2*np.cos(new_lat[ix,iy,iz])*np.cos(new_lat[ix,iy,iz])/(dx*dx)-2*Ra*Ra*np.cos(new_lat[ix,iy,iz])*np.cos(new_lat[ix,iy,iz])*new_f[ix,iy,iz]*new_f[ix,iy,iz]*R[iz]/(dp*dp)
    #                 RR=FRC[ix,iy,iz]-(COE011*GT0[ix-1,iy,iz]+COE111*GT0[ix,iy,iz]+COE211*GT0[ix+1,iy,iz]+COE101*GT0[ix,iy-1,iz]+COE121*GT0[ix,iy+1,iz]+COE110*GT0[ix,iy,iz-1]+COE112*GT0[ix,iy,iz+1])
                    
    #                 GT0[ix,iy,iz]=GT0[ix,iy,iz]+SC/COE111*RR
    
    
    COE101 = np.zeros([nx-2,ny-2,nz-2])
    COE121 = np.zeros([nx-2,ny-2,nz-2])
    COE110 = np.zeros([nx-2,ny-2,nz-2])
    COE112 = np.zeros([nx-2,ny-2,nz-2])
    COE111 = np.zeros([nx-2,ny-2,nz-2])
    RR = np.zeros([nx-2,ny-2,nz-2])
    
    for k1 in range(3000):
        GT1=GT0
        COE011=1/(dx*dx)
        COE211=1/(dx*dx)
        COE101=np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*np.sin(new_lat[1:nx-1,1:ny-1,1:nz-1])/(2*dy)+np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])/(dy*dy)
        COE121=-np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*np.sin(new_lat[1:nx-1,1:ny-1,1:nz-1])/(2*dy)+np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])/(dy*dy)
        COE110=-Ra*Ra*(np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*new_f[1:nx-1,1:ny-1,1:nz-1]*new_f[1:nx-1,1:ny-1,1:nz-1])*(R[:,:,2:nz]-R[:,:,0:nz-2])/(2*dp)/(2*dp)+Ra*Ra*np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*new_f[1:nx-1,1:ny-1,1:nz-1]*new_f[1:nx-1,1:ny-1,1:nz-1]*R[:,:,1:nz-1]/(dp*dp)
        COE112=Ra*Ra*(np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*new_f[1:nx-1,1:ny-1,1:nz-1]*new_f[1:nx-1,1:ny-1,1:nz-1])*(R[:,:,2:nz]-R[:,:,0:nz-2])/(2*dp)/(2*dp)+Ra*Ra*np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*new_f[1:nx-1,1:ny-1,1:nz-1]*new_f[1:nx-1,1:ny-1,1:nz-1]*R[:,:,1:nz-1]/(dp*dp)       
        COE111=-2/(dx*dx)-2*np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])/(dx*dx)-2*Ra*Ra*np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*np.cos(new_lat[1:nx-1,1:ny-1,1:nz-1])*new_f[1:nx-1,1:ny-1,1:nz-1]*new_f[1:nx-1,1:ny-1,1:nz-1]*R[:,:,1:nz-1]/(dp*dp)
        RR=FRC[1:nx-1,1:ny-1,1:nz-1]-(COE011*GT0[0:nx-2,1:ny-1,1:nz-1]+COE111*GT0[1:nx-1,1:ny-1,1:nz-1]+COE211*GT0[2:nx,1:ny-1,1:nz-1]+COE101*GT0[1:nx-1,0:ny-2,1:nz-1]+COE121*GT0[1:nx-1,2:ny,1:nz-1]+COE110*GT0[1:nx-1,1:ny-1,0:nz-2]+COE112*GT0[1:nx-1,1:ny-1,2:nz])
        
        GT0[1:nx-1,1:ny-1,1:nz-1]=GT0[1:nx-1,1:ny-1,1:nz-1]+SC/COE111*RR
        
        
        #Vertical BC
        if forcing_type==2:
            GT0[:,:,0]=GT0[:,:,1];
            GT0[:,:,nz-1]=GT0[:,:,nz-2];
        
        
        if forcing_type==1:
            GT0[:,:,0]=GT0[:,:,1]+R_constant*dp/pp[0]*FRC0[:,:,1];
            GT0[:,:,nz-1]=GT0[:,:,nz-2]-R_constant*dp/pp[nz-1]*FRC0[:,:,nz-1];
    
    
    #     if (all((abs(GT0-GT1))<=CRI))
    #         break;

    #Interpolate back to the raw levels
    GTI=GT1
    for iy in range(ny):
        for ix in range(nx):
            cs = CubicSpline(np.flip(pp),np.flip(GTI[ix,iy,:]))
            GT[ix,iy,:]=np.flip(cs(np.flip(p)))
            
    return GT
    
    
    
    
    
    
    
    
    
    
    
    

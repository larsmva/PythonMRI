#!/usr/bin/env python
# -*- coding: utf-8 -*-
	
def diff(previous ,current ): 
    """
       Returns the seconds difference between to string on the format 
       sequnece-date-time.mgz 

       T1_3D-20082902-101010.mgz   
    """
    import datetime
    p_string = previous[0:-7].split("-")
    c_string = current[0:-7].split("-")
    pdate = datetime.datetime.strptime(p_string[-2]+p_string[-1],'%Y%m%d%H%M%S')
    cdate = datetime.datetime.strptime(c_string[-2]+c_string[-1],'%Y%m%d%H%M%S') 
    dtime = cdate - pdate
	
    return  dtime.days*24*3600+ dtime.seconds




def laplace_operator(u):
    import numpy as np
    Lu = np.zeros(u.shape)
    u_xx = u[2:-1,1:-2,1:-2] - 2*u[1:-2,1:-2,1:-2] +u[0:-3,1:-2,1:-2]
    u_yy = u[1:-2,2:-1,1:-2] - 2*u[1:-2,1:-2,1:-2] +u[1:-2,0:-3,1:-2]
    u_zz = u[1:-2,1:-2,2:-1] - 2*u[1:-2,1:-2,1:-2] +u[1:-2,1:-2,0:-3]
    Lu[1:-2,1:-2,1:-2]  = u_xx + u_yy + u_zz
    return Lu


def Highorder_operator(u):
    import numpy as np
    Lu = np.zeros(u.shape)
    u_xx = u[4:-1,1:-2,1:-2] + 16*u[3:-2,1:-2,1:-2] - 30*u[2:-3,1:-2,1:-2] +16*u[1:-4,1:-2,1:-2] -30*u[0:-5,1:-2,1:-2]
    u_yy = u[1:-2,4:-1,1:-2] + 16*u[1:-2,3:-2,1:-2] - 30*u[1:-2,2:-3,1:-2] +16*u[1:-2,1:-4,1:-2] -30*u[1:-2,0:-5,1:-2]
    u_zz = u[1:-2,1:-2,4:-1] + 16*u[1:-2,1:-2,3:-2] - 30*u[1:-2,1:-2,2:-3] +16*u[1:-2,1:-2,1:-4] -30*u[1:-2,1:-2,0:-5]
    Lu[2:-3,1:-2,1:-2]+=u_xx/12.
    Lu[1:-2,2:-3,1:-2]+=u_yy/12.
    Lu[1:-2,1:-2,2:-3]+=u_zz/12.
    return Lu


def compute_D(folder, segmentation) :
    import os
    import numpy as np 

    import nibabel as nb
    import scipy.optimize as opt 
    files =[] 
    segdata = nb.load(segmentation).get_data()
    label = 1
    save = os.path.dirname(os.path.dirname(files[0]))+"/D"
    for file in sorted(os.listdir(folder)):
        if file.endswith("-C.mgz"):
           files.append(folder+"/"+file)

    for no in range(1,len(files)):
        vox2ras = nb.load(files[no]).get_header().get_vox2ras()
	u0 =nb.load(files[no-1]).get_data()
	u1 =nb.load(files[no]).get_data()

        ut= np.zeros(u0.shape)
        Lu= np.zeros(u0.shape)
        ut[1:-2,1:-2,1:-2] = (u1[1:-2,1:-2,1:-2]-u0[1:-2,1:-2,1:-2] )
                
               
                #Lu =laplace_operator(u1)     
        Lu = Highorder_operator(u1)
        ut[segdata!=label]=0
        Lu[segdata!=label]=0

        ut[ut<0]=0
        Lu[Lu<0]=0
        #ut[Lu<0]=0
        #Lu[ut<0]=0

        #dt1=diff(files[no],files[no+1]) 

        dt=diff(files[no-1],files[no]) 

        D = ut/(dt*Lu)
                
        m  = ut.flatten()
        Lm = Lu.flatten()
        def func(D) :
            return  sum ( (m- dt*D*Lm)**2 ) 

        print opt.minimize_scalar(func)

        D[np.isnan(D)]=0
        D[np.isinf(D)]=0

        Dcoef=D*1e6

        img2 = nb.MGHImage(Dcoef.astype("float32"),vox2ras)
	                       
        nb.save(img2,save+os.path.basename(files[0]))



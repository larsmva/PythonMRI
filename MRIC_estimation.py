#!/usr/bin/env python
# -*- coding: utf-8 -*-





def MRIC_estimation(k1,k2,G,theta,I): 
    """#!/usr/bin/env python
# -*- coding: utf-8 -*-
    Description:
        Solver for F(c) = 0  for c in <0,c_max>
                      
        ==============	 ==============================================================
        Argument                Explanation
        ==============	 ==============================================================
        k1		 Defined as TR*r1
        k2		 Defind as TE*r2
        G		 Defined exp( -TR*R1)
        theta            The flip angle
        I		 The baseline procentage intensity increase 
    """
    import numpy as np
    import scipy.optimize as opt

    aux0 = np.cos(np.deg2rad(theta)) 
    aux1 = 1.-aux0
    aux2 = 1.+aux0

    def Ir_GRE(c) : 
        return (1.-aux0*G)*(1.-G*np.exp(-k1*c))*np.exp(-k2*c)/((1.-aux0*G*np.exp(-k1*c))*(1.-G)
         
    x = ( k1*aux1 + k2*aux2  + np.sqrt(k1**2*aux1**2 + 2*k1*k2*aux1*aux2 + k2**2*aux1**2))/(2*k2) # if /aux0
    c_max = np.log(x/G)/k1  	
       
    Im = Ir_GRE(c_max) 

    if I>Im: 
       return c_max*1000
    if I<1.0:
       return 0
    
    def F(c) : 
	return (1.-G*np.exp(-k1*c))*np.exp(-k2*c)/(1.-aux0*G*np.exp(-k1*c))  -I*(1.-G)/(1.-aux0*G) 
     

    c =  opt.brentq(F,0.0,c_max)
	
    return c*1000



def concentration_image(path2mri, path2seg ) :

    import nibabel as nb
    from nibabel.affines import apply_affine
    import numpy.linalg as npl
    import numpy as np
    
    idata  = nb.load(path2mri).get_data()

    vox2ras = nb.load(path2mri).get_header().get_vox2ras()
    I_r   = idata.reshape(1,-1)[0].astype("float32")
    iseg   = nb.load(path2seg).get_data()
    Seg   = iseg.reshape(1,-1)[0].astype("int8")
   
    TR    = 5.12*10**-3
    TE    = 2.29*10**-3
    theta = 8.0
    r1,r2 = 3.2 , 3.9
    k1 = TR*r1
    k2  = TE*r2  

    G1 = np.exp(-TR/3.817)
    G2 = np.exp(-TR/1.080)
    G3 = np.exp(-TR/1.820)

    G=[G1,G2,G3]
   
    numv = len(I_r)
    c = np.zeros(numv)
    for i in range(numv) :

        if Seg[i] == 0 :
           c[i]=0
        else :
           c[i] =  mric_brentq(k1,k2,G[Seg[i]-1],theta,I_r[i]/100+1) 

    img2 = nb.MGHImage(c.reshape(idata.shape).astype("float32") ,vox2ras)
    nb.save(img2,path2mri.split(".")[0]+"-C.mgz")
   

def concentration_images(folder, path2seg ) :
    import os
    for file in sorted(os.listdir(folder)):
        if file.endswith(".mgz"):
           concentration_image(folder+"/"+file, path2seg ) 
  

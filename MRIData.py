#!/usr/bin/env python
# -*- coding: utf-8 -*-




def neighboor_values( data,i,j,k):
		import numpy as np
		temp = data[i-2:i+3,j-2:j+3,k-2:k+3].reshape(1,-1)[0]
		v = np.sort(temp)
		return sum(v[-6:-1])/len(v[-6:-1])


def get_vector(path2mri_file,function_space):
		"""
		==============	 ==============================================================
		Argument                Explanation
		==============	 ==============================================================
		path2_mri_file	The path to the principal eigenvector file with extension mgz.
		function_space	The function space of the mesh.		
		"""
		import nibabel
		from nibabel.affines import apply_affine
		import numpy.linalg as npl
		import numpy as np

		img= nibabel.load(path2mri_file) 
		inv_aff = npl.inv ( img.get_header().get_vox2ras_tkr() )
		data = img.get_data()
	
		xyz = function_space.tabulate_dof_coordinates().reshape((function_space.dim(),-1))[0::3] # there are 3 dofs pr vertex

		i,j,k = np.rint(apply_affine(inv_aff,xyz).T)
		
		
		# data.item is a vector, thus
		# we get a 3xn matrix
		# this reshaped to a 1x3n array 
		# with structure
		# v1(1) v1(2) v1(3) v2(1) v2(2) v2(3) ... vn(1) vn(2) vn(3) 
		
		return np.array(data[i,j,k]).reshape(-1)


def get_FD_values(path2mri_file,function_space):
		"""
		==============	 ==============================================================
		Argument                Explanation
		==============	 ==============================================================
		path2_mri_file	The path to the Neural density file with extension mgz.
		function_space	The function space of the mesh.		
		"""
		import nibabel
		from nibabel.affines import apply_affine
		import numpy.linalg as npl
		import numpy as np

		img= nibabel.load(path2mri_file) 
		inv_aff = npl.inv ( img.get_header().get_vox2ras_tkr() )
		data = img.get_data()

		xyz = function_space.tabulate_dof_coordinates().reshape((function_space.dim(),-1)) # there are 3 dofs pr vertex
	
		i,j,k = np.rint(apply_affine(inv_aff,xyz).T)
		
		
		# data.item is a vector, thus
		# we get a 3xn matrix
		# this reshaped to a 1x3n array 
		# with structure
		# v1(1) v1(2) v1(3) v2(1) v2(2) v2(3) ... vn(1) vn(2) vn(3) 
		
		return np.array(data[i,j,k,0]).reshape(-1)


def get_eigenvectorFD(eigenvector_mgz, ND_mgz,function_space):
		"""
		==============	  ==============================================================
		Argument                Explanation
		==============	  ==============================================================
		eigenvector_mgz   
                ND_mgz
		function_space	  The function space of the mesh.		
		"""
		import nibabel
		from nibabel.affines import apply_affine
		import numpy.linalg as npl
		import numpy as np

		eigenvectors = get_vector(eigenvector_mgz,function_space)
		FD = get_FD_values(ND_mgz,function_space)

		return FD*eigenvectors
	


def get_4D(path2mri_file,function_space):
		"""
		==============	 ==============================================================
		Argument                Explanation
		==============	 ==============================================================
		path2_mri_file	The path to the MRI file with extension mgz
		function_space	The function space of the mesh.		

                return          The returned array is of nxm with a structure [ v1 , v2 ... ,vm]
		"""
		import nibabel
		from nibabel.affines import apply_affine
		import numpy.linalg as npl
		import numpy as np
                
		img= nibabel.load(path2mri_file) 
		inv_aff = npl.inv ( img.get_header().get_vox2ras_tkr() ) 

                data = img.get_data() 
                
	        dim1,dim2,dim3,dim4 = data.shape 
	
		xyz = function_space.tabulate_dof_coordinates().reshape((function_space.dim(),-1))


	        i,j,k = np.rint(apply_affine(inv_aff,xyz).T)
        
             
		values = np.zeros((len(i),dim4))

                for m in range(dim4):
			for n in range(len(i)):
				values[n,m]= neighboor_values(data[:,:,:,m],i[n],j[n],k[n]) 
	     

		# this reshaped to a nxm array 
		# with structure
		# [v1,v2,v3,v4] 
		return values




def get_MRIData(path2mri_file,function_space):
		"""
                Comments: 
                        Scalar valeus are computed by a local average of max values
		==============	 ==============================================================
		Argument                Explanation
		==============	 ==============================================================
		path2_mri_file	The path to the MRI file with extension mgz. typically orig.mgz
		function_space	The function space of the mesh.		 	
		"""
		import nibabel
		from nibabel.affines import apply_affine
		import numpy.linalg as npl
		import numpy as np
                import dolfin as df


                # If the element is VectorElement then it will be stored as 4D object
                if isinstance(function_space.ufl_element(),df.VectorElement):
                   return get_Vector(path2mri_file,function_space)

		img= nibabel.load(path2mri_file) 
		inv_aff = npl.inv ( img.get_header().get_vox2ras_tkr() )
                
                if  (function_space.ufl_element()._family=='Discontinuous Lagrange'):
                    print "something"

		data = img.get_data()

                # For scalar voxel values at different tiimepoints
                if len(data.shape)==4:
                    return get_4D(path2mri_file,function_space)

                else :
			xyz = function_space.tabulate_dof_coordinates().reshape((function_space.dim(),-1))

			i,j,k = np.rint(apply_affine(inv_aff,xyz).T)
	
			values = np.zeros(len(i))
		   
			for n in range(len(i)):
				values[n]= data[i[n],j[n],k[n]] 
	     
			return values


def write_hdf5( path2mri, meshfile, savefile , space="CG", order=1):

		import dolfin as df
		mesh = df.Mesh()
		if meshfile.endswith(".xml"):
		   mesh = df.Mesh(meshfile)
		elif meshfile.endswith(".xdmf"):
		   xdmf = df.XDMFFile(mesh.mpi_comm(),meshfile)
		   xdmf.read(mesh)
		   xdmf.close()
			 
		V = df.FunctionSpace(mesh,space,order)

		u = df.Function(V)	    

		u.vector()[:]= get_MRIData(path2mri,V)	    

		hdf5 = df.HDF5File(mesh.mpi_comm(),savefile+".h5", "w")	    
		hdf5.write(u,"/"+"kontrast") 	    
		hdf5.close() 	    
                df.File(savefile+".pvd") << u

if __name__ =='__main__':
	import argparse
        import os
	parser = argparse.ArgumentParser()
	parser.add_argument('--folder',type=str) 
        parser.add_argument('--mesh', type=str) 
        parser.add_argument('--save', type=str) 

        Z = parser.parse_args() 
	
      
        
        for file in sorted(os.listdir(Z.folder)):
                 if file.endswith(".mgz"):
                    write_hdf5(Z.folder+"/"+file, Z.mesh ,Z.save+"/"+file[0:-4], space="CG", order=1) 





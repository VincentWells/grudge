from meshmode.array_context import PyOpenCLArrayContext, make_loopy_program
import loopy as lp
import pyopencl
import pyopencl.array as cla
import loopy_dg_kernels as dgk
import numpy as np

ctof_knl = lp.make_copy_kernel("f,f", old_dim_tags="c,c")
ftoc_knl = lp.make_copy_kernel("c,c", old_dim_tags="f,f")

# Really this is more of an Nvidia array context probably
class GrudgeArrayContext(PyOpenCLArrayContext):

    def __init__(self, queue, allocator=None):
        super().__init__(queue, allocator=allocator)

    def empty(self, shape, dtype):
        return cla.empty(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator, order='F')

    def zeros(self, shape, dtype):
        return cla.zeros(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator, order='F')

    '''
    def from_numpy(self, np_array: np.ndarray):
        # Should intercept this for the dof array
        # and make it return a fortran style array 
        # Creation of the field seems to not pass
        # through here though.
        cl_array_c = super().from_numpy(np_array)
        #print(cl_array_c.shape)
        return cl_array_c
        #cq = cl_arry.queue # Is this necessary?
        #_,(cl_array_f,) = ctof_knl(cq, input=cl_array_c)
        #return cl_array_f 
    '''

    def call_loopy(self, program, **kwargs):

        #print("Program: " + program.name)
        if program.name == "opt_diff":
            diff_mat = kwargs["diff_mat"]
            result = kwargs["result"]
            vec = kwargs["vec"]

            # Create input array
            cq = vec.queue
            dtp = vec.dtype

            # Esto no deberia hacerse aqui.
            #_,(inArg,) = ctof_knl(cq, input=vec)
            
            # Treat as c array, can do this to use c-format diff function
            # np.array(A, format="F").flatten() == np.array(A.T, format="C").flatten()
            #inArg.shape = (inArg.shape[1], inArg.shape[0])
            #inArg.strides = cla._make_strides(vec.dtype.itemsize, inArg.shape, "c")
            #outShape = inArg.shape #(inArg.shape[1], inArg.shape[0])

            argDict = { "result1": cla.Array(cq, vec.shape, dtp, order="f"),
                        "result2": cla.Array(cq, vec.shape, dtp, order="f"),
                        "result3": cla.Array(cq, vec.shape, dtp, order="f"),
                        "vec": vec,
                        "mat1": diff_mat[0],
                        "mat2": diff_mat[1],
                        "mat3": diff_mat[2] }
              
            super().call_loopy(program, **argDict)

            result = kwargs["result"]

            # Treat as fortran style array again
            #for i, entry in enumerate(["result1", "result2", "result3"]):
            #    argDict[entry].shape = (argDict[entry].shape[1], argDict[entry].shape[0])
            #    argDict[entry].strides = cla._make_strides(argDict[entry].dtype.itemsize, argDict[entry].shape, "f")
                # This should be unnecessary
                # Il est necessaire pour le moment a cause du "ctof" d'ici. 
                #ftoc_knl(cq, input=argDict[entry], output=result[i])

        else:
            result = super().call_loopy(program,**kwargs)

        return result

    #memoize_method
    def _get_scalar_func_loopy_program(self, name, nargs, naxes):
        prog = super()._get_scalar_func_loopy_program(name, nargs, naxes)
        for arg in prog.args:
            if type(arg) == lp.ArrayArg:
                arg.tags = "dof_array"
        return prog

    
    # Side note: the meaning of thawed and frozen seem counterintuitive to me.
    def thaw(self, array):
        thawed = super().thaw(array)
        if getattr(array, "tags", None) == "dof_array":           
            cq = thawed.queue
            _, (out,) = ctof_knl(cq, input=thawed)
            thawed = out
            # May or may not be needed
            thawed.tags = "dof_array"
        return thawed

    #@memoize_method
    def transform_loopy_program(self, program):

        #print(program.name)
        for arg in program.args:
            if arg.tags == "dof_array":
                program = lp.tag_array_axes(program, arg.name, "f,f")
            elif arg.tags == "mult_dof_array":
                program = lp.tag_array_axes(program, arg.name, "sep,f,f")        
            elif arg.tags == "face_dof_array":
                program = lp.tag_array_axes(program, arg.name, "N1,N0,N2")        

        if program.name == "opt_diff":
            # TODO: Dynamically determine device id, don't hardcode path to transform.hjson.
            # Also get pn from program
            filename = "/home/njchris2/Workspace/nick/loopy_dg_kernels/transform.hjson"
            deviceID = "NVIDIA Titan V"
            pn = 3

            transformations = dgk.loadTransformationsFromFile(filename, deviceID, pn)            
            program = dgk.applyTransformationList(program, transformations)
        else:
            program = super().transform_loopy_program(program)

        return program

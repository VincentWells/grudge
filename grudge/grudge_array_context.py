from meshmode.array_context import PyOpenCLArrayContext, make_loopy_program
from meshmode.dof_array import DOFTag
from grudge.execution import VecDOFTag, VecOpDOFTag, FaceDOFTag
import loopy as lp
import pyopencl
import pyopencl.array as cla
import loopy_dg_kernels as dgk
import numpy as np

ctof_knl = lp.make_copy_kernel("f,f", old_dim_tags="c,c")
ftoc_knl = lp.make_copy_kernel("c,c", old_dim_tags="f,f")

class GrudgeArrayContext(PyOpenCLArrayContext):

    def empty(self, shape, dtype):
        return cla.empty(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator, order='F')

    def zeros(self, shape, dtype):
        return cla.zeros(self.queue, shape=shape, dtype=dtype,
                allocator=self.allocator, order='F')

    #@memoize_method
    def _get_scalar_func_loopy_program(self, name, nargs, naxes):
        prog = super()._get_scalar_func_loopy_program(name, nargs, naxes)
        for arg in prog.args:
            if type(arg) == lp.ArrayArg:
                arg.tags = DOFTag()
        return prog

    def thaw(self, array):
        thawed = super().thaw(array)
        if type(getattr(array, "tags", None)) == DOFTag:           
            cq = thawed.queue
            _, (out,) = ctof_knl(cq, input=thawed)
            thawed = out
            # May or may not be needed
            #thawed.tags = "dof_array"
        return thawed

    #@memoize_method
    def transform_loopy_program(self, program):

        for arg in program.args:
            if isinstance(arg.tags, DOFTag):
                program = lp.tag_array_axes(program, arg.name, "f,f")
            elif isinstance(arg.tags, VecDOFTag):
                program = lp.tag_array_axes(program, arg.name, "sep,f,f")        
            #elif isinstance(arg.tags, VecOpDOFTag):
            #    program = lp.tag_array_axes(program, arg.name, "sep,c,c")
            elif isinstance(arg.tags, FaceDOFTag):
                program = lp.tag_array_axes(program, arg.name, "N1,N0,N2")        

        if program.name == "opt_diff":
            # TODO: Dynamically determine device id, don't hardcode path to transform.hjson.
            # Also get pn from program
            filename = "/home/njchris2/Workspace/nick/loopy_dg_kernels/transform.hjson"
            deviceID = "NVIDIA Titan V"
            pn = 4

            transformations = dgk.loadTransformationsFromFile(filename, deviceID, pn)            
            program = dgk.applyTransformationList(program, transformations)
        else:
            program = super().transform_loopy_program(program)

        return program

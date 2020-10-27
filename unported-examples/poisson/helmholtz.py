__copyright__ = "Copyright (C) 2007 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
import numpy
import numpy.linalg as la
from grudge.tools import Reflection, Rotation



class ResidualPrinter:
    def __init__(self, compute_resid=None):
        self.count = 0
        self.compute_resid = compute_resid

    def __call__(self, cur_sol):
        import sys

        if cur_sol is not None:
            if self.count % 20 == 0:
                sys.stdout.write("IT %8d %g                 \r" % (
                    self.count, la.norm(self.compute_resid(cur_sol))))
        else:
            sys.stdout.write("IT %8d                    \r" % self.count)
        self.count += 1
        sys.stdout.flush()




def main(write_output=True):
    from grudge.data import GivenFunction, ConstantGivenFunction

    from grudge.backends import guess_run_context
    rcon = guess_run_context()

    dim = 2

    def boundary_tagger(fvi, el, fn, points):
        from math import atan2, pi
        normal = el.face_normals[fn]
        if -90/180*pi < atan2(normal[1], normal[0]) < 90/180*pi:
            return ["neumann"]
        else:
            return ["dirichlet"]

    def dirichlet_boundary_tagger(fvi, el, fn, points):
            return ["dirichlet"]

    if dim == 2:
        if rcon.is_head_rank:
            from grudge.mesh.generator import make_disk_mesh
            mesh = make_disk_mesh(r=0.5,
                    boundary_tagger=dirichlet_boundary_tagger,
                    max_area=1e-3)
    elif dim == 3:
        if rcon.is_head_rank:
            from grudge.mesh.generator import make_ball_mesh
            mesh = make_ball_mesh(max_volume=0.0001,
                    boundary_tagger=lambda fvi, el, fn, points:
                    ["dirichlet"])
    else:
        raise RuntimeError("bad number of dimensions")

    if rcon.is_head_rank:
        print("%d elements" % len(mesh.elements))
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    discr = rcon.make_discretization(mesh_data, order=5,
            debug=[])

    def dirichlet_bc(x, el):
        from math import sin
        return sin(10*x[0])

    def rhs_c(x, el):
        if la.norm(x) < 0.1:
            return 1000
        else:
            return 0

    def my_diff_tensor():
        result = numpy.eye(dim)
        result[0,0] = 0.1
        return result

    try:
        from grudge.models.poisson import (
                PoissonOperator,
                HelmholtzOperator)
        from grudge.second_order import \
                IPDGSecondDerivative, LDGSecondDerivative, \
                StabilizedCentralSecondDerivative

        k = 1

        from grudge.mesh import BTAG_NONE, BTAG_ALL
        op = HelmholtzOperator(k, discr.dimensions,
                #diffusion_tensor=my_diff_tensor(),

                #dirichlet_tag="dirichlet",
                #neumann_tag="neumann",

                dirichlet_tag=BTAG_ALL,
                neumann_tag=BTAG_NONE,

                #dirichlet_tag=BTAG_ALL,
                #neumann_tag=BTAG_NONE,

                #dirichlet_bc=GivenFunction(dirichlet_bc),
                dirichlet_bc=ConstantGivenFunction(0),
                neumann_bc=ConstantGivenFunction(-10),

                scheme=StabilizedCentralSecondDerivative(),
                #scheme=LDGSecondDerivative(),
                #scheme=IPDGSecondDerivative(),
                )
        bound_op = op.bind(discr)

        if False:
            from grudge.iterative import parallel_cg
            u = -parallel_cg(rcon, -bound_op,
                    bound_op.prepare_rhs(discr.interpolate_volume_function(rhs_c)),
                    debug=20, tol=5e-4,
                    dot=discr.nodewise_dot_product,
                    x=discr.volume_zeros())
        else:
            rhs = bound_op.prepare_rhs(discr.interpolate_volume_function(rhs_c))
            def compute_resid(x):
                return bound_op(x)-rhs

            from scipy.sparse.linalg import minres, LinearOperator
            u, info = minres(
                    LinearOperator(
                        (len(discr), len(discr)),
                        matvec=bound_op, dtype=bound_op.dtype),
                    rhs,
                    callback=ResidualPrinter(compute_resid),
                    tol=1e-5)
            print()
            if info != 0:
                raise RuntimeError("gmres reported error %d" % info)
            print("finished gmres")

            print(la.norm(bound_op(u)-rhs)/la.norm(rhs))

        if write_output:
            from grudge.visualization import SiloVisualizer, VtkVisualizer
            vis = VtkVisualizer(discr, rcon)
            visf = vis.make_file("fld")
            vis.add_data(visf, [ ("sol", discr.convert_volume(u, kind="numpy")), ])
            visf.close()
    finally:
        discr.close()





if __name__ == "__main__":
    main()

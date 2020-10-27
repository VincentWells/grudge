__copyright__ = "Copyright (C) 2008 Andreas Kloeckner"

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




class SteadyShearFlow:
    def __init__(self):
        self.gamma = 1.5
        self.mu = 0.01
        self.prandtl = 0.72
        self.spec_gas_const = 287.1

    def __call__(self, t, x_vec):
        # JSH/TW Nodal DG Methods, p.326

        rho = numpy.ones_like(x_vec[0])
        rho_u = x_vec[1] * x_vec[1]
        rho_v = numpy.zeros_like(x_vec[0])
        e = (2 * self.mu * x_vec[0] + 10) / (self.gamma - 1) + x_vec[1]**4 / 2

        from grudge.tools import join_fields
        return join_fields(rho, e, rho_u, rho_v)

    def properties(self):
        return(self.gamma, self.mu, self.prandtl, self.spec_gas_const)

    def volume_interpolant(self, t, discr):
        return discr.convert_volume(
                        self(t, discr.nodes.T
                            .astype(discr.default_scalar_type)),
                        kind=discr.compute_kind)

    def boundary_interpolant(self, t, discr, tag):
        result = discr.convert_boundary(
                        self(t, discr.get_boundary(tag).nodes.T
                            .astype(discr.default_scalar_type)),
                        tag=tag, kind=discr.compute_kind)
        return result




def main():
    from grudge.backends import guess_run_context
    rcon = guess_run_context(
    #["cuda"]
    )

    from grudge.tools import EOCRecorder, to_obj_array
    eoc_rec = EOCRecorder()

    def boundary_tagger(vertices, el, face_nr, all_v):
        return ["inflow"]

    if rcon.is_head_rank:
        from grudge.mesh import make_rect_mesh, \
                               make_centered_regular_rect_mesh
        #mesh = make_rect_mesh((0,0), (10,1), max_area=0.01)
        refine = 1
        mesh = make_centered_regular_rect_mesh((0,0), (10,1), n=(20,4),
                            #periodicity=(True, False),
                            post_refine_factor=refine,
                            boundary_tagger=boundary_tagger)
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    for order in [3]:
        discr = rcon.make_discretization(mesh_data, order=order,
                        default_scalar_type=numpy.float64)

        from grudge.visualization import SiloVisualizer, VtkVisualizer
        #vis = VtkVisualizer(discr, rcon, "shearflow-%d" % order)
        vis = SiloVisualizer(discr, rcon)

        shearflow = SteadyShearFlow()
        fields = shearflow.volume_interpolant(0, discr)
        gamma, mu, prandtl, spec_gas_const = shearflow.properties()

        from grudge.models.gas_dynamics import GasDynamicsOperator
        op = GasDynamicsOperator(dimensions=2, gamma=gamma, mu=mu,
                prandtl=prandtl, spec_gas_const=spec_gas_const,
                bc_inflow=shearflow, bc_outflow=shearflow, bc_noslip=shearflow,
                inflow_tag="inflow", outflow_tag="outflow", noslip_tag="noslip")

        navierstokes_ex = op.bind(discr)

        max_eigval = [0]
        def rhs(t, q):
            ode_rhs, speed = navierstokes_ex(t, q)
            max_eigval[0] = speed
            return ode_rhs

        # needed to get first estimate of maximum eigenvalue
        rhs(0, fields)

        if rcon.is_head_rank:
            print("---------------------------------------------")
            print("order %d" % order)
            print("---------------------------------------------")
            print("#elements=", len(mesh.elements))

        from grudge.timestep import RK4TimeStepper
        stepper = RK4TimeStepper()

        # diagnostics setup ---------------------------------------------------
        from logpyle import LogManager, add_general_quantities, \
                add_simulation_quantities, add_run_info

        logmgr = LogManager("navierstokes-cpu-%d-%d.dat" % (order, refine),
                            "w", rcon.communicator)
        add_run_info(logmgr)
        add_general_quantities(logmgr)
        add_simulation_quantities(logmgr)
        discr.add_instrumentation(logmgr)
        stepper.add_instrumentation(logmgr)

        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

        # timestep loop -------------------------------------------------------
        try:
            from grudge.timestep import times_and_steps
            step_it = times_and_steps(
                    final_time=0.3,
                    #max_steps=500,
                    logmgr=logmgr,
                    max_dt_getter=lambda t: op.estimate_timestep(discr,
                        stepper=stepper, t=t, max_eigenvalue=max_eigval[0]))

            for step, t, dt in step_it:
                if step % 10 == 0:
                #if False:
                    visf = vis.make_file("shearflow-%d-%04d" % (order, step))

                    #true_fields = shearflow.volume_interpolant(t, discr)

                    from pyvisfile.silo import DB_VARTYPE_VECTOR
                    vis.add_data(visf,
                            [
                                ("rho", discr.convert_volume(op.rho(fields), kind="numpy")),
                                ("e", discr.convert_volume(op.e(fields), kind="numpy")),
                                ("rho_u", discr.convert_volume(op.rho_u(fields), kind="numpy")),
                                ("u", discr.convert_volume(op.u(fields), kind="numpy")),

                                #("true_rho", discr.convert_volume(op.rho(true_fields), kind="numpy")),
                                #("true_e", discr.convert_volume(op.e(true_fields), kind="numpy")),
                                #("true_rho_u", discr.convert_volume(op.rho_u(true_fields), kind="numpy")),
                                #("true_u", discr.convert_volume(op.u(true_fields), kind="numpy")),
                                ],
                            expressions=[
                                #("diff_rho", "rho-true_rho"),
                                #("diff_e", "e-true_e"),
                                #("diff_rho_u", "rho_u-true_rho_u", DB_VARTYPE_VECTOR),

                                ("p", "0.4*(e- 0.5*(rho_u*u))"),
                                ],
                            time=t, step=step
                            )
                    visf.close()

                fields = stepper(fields, t, dt, rhs)

            true_fields = shearflow.volume_interpolant(t, discr)
            l2_error = discr.norm(op.u(fields)-op.u(true_fields))
            eoc_rec.add_data_point(order, l2_error)
            print()
            print(eoc_rec.pretty_print("P.Deg.", "L2 Error"))

            logmgr.set_constant("l2_error", l2_error)

        finally:
            vis.close()
            logmgr.save()
            discr.close()

if __name__ == "__main__":
    main()

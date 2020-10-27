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
from six.moves import range




def make_nacamesh():
    def round_trip_connect(seq):
        result = []
        for i in range(len(seq)):
            result.append((i, (i+1)%len(seq)))
        return result

    pt_back = numpy.array([1,0])

    #def max_area(pt):
        #max_area_front = 1e-2*la.norm(pt)**2 + 1e-5
        #max_area_back = 1e-2*la.norm(pt-pt_back)**2 + 1e-4
        #return min(max_area_front, max_area_back)

    def max_area(pt):
        x = pt[0]

        if x < 0:
            return 1e-2*la.norm(pt)**2 + 1e-5
        elif x > 1:
            return 1e-2*la.norm(pt-pt_back)**2 + 1e-5
        else:
            return 1e-2*pt[1]**2 + 1e-5

    def needs_refinement(vertices, area):
        barycenter =  sum(numpy.array(v) for v in vertices)/3
        return bool(area > max_area(barycenter))

    from meshpy.naca import get_naca_points
    points = get_naca_points(naca_digits="2412", number_of_points=80)

    from meshpy.geometry import GeometryBuilder, Marker
    from meshpy.triangle import write_gnuplot_mesh

    profile_marker = Marker.FIRST_USER_MARKER
    builder = GeometryBuilder()
    builder.add_geometry(points=points,
            facets=round_trip_connect(points),
            facet_markers=profile_marker)
    builder.wrap_in_box(4, (10, 8))

    from meshpy.triangle import MeshInfo, build
    mi = MeshInfo()
    builder.set(mi)
    mi.set_holes([builder.center()])

    mesh = build(mi, refinement_func=needs_refinement,
            #allow_boundary_steiner=False,
            generate_faces=True)

    write_gnuplot_mesh("mesh.dat", mesh)

    print("%d elements" % len(mesh.elements))

    fvi2fm = mesh.face_vertex_indices_to_face_marker

    face_marker_to_tag = {
            profile_marker: "noslip",
            Marker.MINUS_X: "inflow",
            Marker.PLUS_X: "outflow",
            Marker.MINUS_Y: "inflow",
            Marker.PLUS_Y: "inflow"
            #Marker.MINUS_Y: "minus_y",
            #Marker.PLUS_Y: "plus_y"
            }

    def bdry_tagger(fvi, el, fn, all_v):
        face_marker = fvi2fm[fvi]
        return [face_marker_to_tag[face_marker]]

    from grudge.mesh import make_conformal_mesh_ext

    vertices = numpy.asarray(mesh.points, order="C")
    from grudge.mesh.element import Triangle
    return make_conformal_mesh_ext(
            vertices,
            [Triangle(i, el_idx, vertices)
                for i, el_idx in enumerate(mesh.elements)],
            bdry_tagger,
            #periodicity=[None, ("minus_y", "plus_y")]
            )




def main():
    from grudge.backends import guess_run_context
    rcon = guess_run_context()

    if rcon.is_head_rank:
        mesh = make_nacamesh()
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    from pytools import add_python_path_relative_to_script
    add_python_path_relative_to_script("..")

    for order in [4]:
        from gas_dynamics_initials import UniformMachFlow
        uniform_flow = UniformMachFlow()

        from grudge.models.gas_dynamics import GasDynamicsOperator, GammaLawEOS
        op = GasDynamicsOperator(dimensions=2,
                equation_of_state=GammaLawEOS(uniform_flow.gamma),
                prandtl=uniform_flow.prandtl,
                spec_gas_const=uniform_flow.spec_gas_const, mu=uniform_flow.mu,
                bc_inflow=uniform_flow, bc_outflow=uniform_flow, bc_noslip=uniform_flow,
                inflow_tag="inflow", outflow_tag="outflow", noslip_tag="noslip")

        discr = rcon.make_discretization(mesh_data, order=order,
                        debug=[
                            "cuda_no_plan",
                            #"cuda_dump_kernels",
                            #"dump_optemplate_stages",
                            #"dump_dataflow_graph",
                            #"print_op_code"
                            ],
                        default_scalar_type=numpy.float32,
                        tune_for=op.sym_operator())

        from grudge.visualization import SiloVisualizer, VtkVisualizer
        #vis = VtkVisualizer(discr, rcon, "shearflow-%d" % order)
        vis = SiloVisualizer(discr, rcon)

        fields = uniform_flow.volume_interpolant(0, discr)

        navierstokes_ex = op.bind(discr)

        max_eigval = [0]
        def rhs(t, q):
            ode_rhs, speed = navierstokes_ex(t, q)
            max_eigval[0] = speed
            return ode_rhs
        rhs(0, fields)

        if rcon.is_head_rank:
            print("---------------------------------------------")
            print("order %d" % order)
            print("---------------------------------------------")
            print("#elements=", len(mesh.elements))

        from grudge.timestep.runge_kutta import \
                ODE23TimeStepper, LSRK4TimeStepper
        stepper = ODE23TimeStepper(dtype=discr.default_scalar_type,
                rtol=1e-6,
                vector_primitive_factory=discr.get_vector_primitive_factory())
        #stepper = LSRK4TimeStepper(dtype=discr.default_scalar_type)

        # diagnostics setup ---------------------------------------------------
        from logpyle import LogManager, add_general_quantities, \
                add_simulation_quantities, add_run_info

        logmgr = LogManager("cns-naca-%d.dat" % order, "w", rcon.communicator)

        add_run_info(logmgr)
        add_general_quantities(logmgr)
        add_simulation_quantities(logmgr)
        discr.add_instrumentation(logmgr)
        stepper.add_instrumentation(logmgr)

        from logpyle import LogQuantity
        class ChangeSinceLastStep(LogQuantity):
            """Records the change of a variable between a time step and the previous
               one"""

            def __init__(self, name="change"):
                LogQuantity.__init__(self, name, "1", "Change since last time step")

                self.old_fields = 0

            def __call__(self):
                result = discr.norm(fields - self.old_fields)
                self.old_fields = fields
                return result

        #logmgr.add_quantity(ChangeSinceLastStep())

        # filter setup-------------------------------------------------------------
        from grudge.discretization import Filter, ExponentialFilterResponseFunction
        mode_filter = Filter(discr,
                ExponentialFilterResponseFunction(min_amplification=0.9,order=4))
        # timestep loop -------------------------------------------------------

        logmgr.add_watches(["step.max", "t_sim.max", "t_step.max"])

        try:
            from grudge.timestep import times_and_steps
            step_it = times_and_steps(
                    final_time=200,
                    #max_steps=500,
                    logmgr=logmgr,
                    max_dt_getter=lambda t: next_dt,
                    taken_dt_getter=lambda: taken_dt)

            model_stepper = LSRK4TimeStepper()
            next_dt = op.estimate_timestep(discr,
                    stepper=model_stepper, t=0,
                    max_eigenvalue=max_eigval[0])

            for step, t, dt in step_it:
                if step % 10 == 0:
                    visf = vis.make_file("naca-%d-%06d" % (order, step))

                    from pyvisfile.silo import DB_VARTYPE_VECTOR
                    vis.add_data(visf,
                            [
                                ("rho", discr.convert_volume(op.rho(fields), kind="numpy")),
                                ("e", discr.convert_volume(op.e(fields), kind="numpy")),
                                ("rho_u", discr.convert_volume(op.rho_u(fields), kind="numpy")),
                                ("u", discr.convert_volume(op.u(fields), kind="numpy")),

                                #("true_rho", op.rho(true_fields)),
                                #("true_e", op.e(true_fields)),
                                #("true_rho_u", op.rho_u(true_fields)),
                                #("true_u", op.u(true_fields)),

                                #("rhs_rho", discr.convert_volume(op.rho(rhs_fields), kind="numpy")),
                                #("rhs_e", discr.convert_volume(op.e(rhs_fields), kind="numpy")),
                                #("rhs_rho_u", discr.convert_volume(op.rho_u(rhs_fields), kind="numpy")),
                                ],
                            expressions=[
                                #("diff_rho", "rho-true_rho"),
                                #("diff_e", "e-true_e"),
                                #("diff_rho_u", "rho_u-true_rho_u", DB_VARTYPE_VECTOR),

                                ("p", "(0.4)*(e- 0.5*(rho_u*u))"),
                                ],
                            time=t, step=step
                            )
                    visf.close()

                fields, t, taken_dt, next_dt = stepper(fields, t, dt, rhs)
                fields = mode_filter(fields)

        finally:
            vis.close()
            logmgr.save()
            discr.close()

if __name__ == "__main__":
    main()

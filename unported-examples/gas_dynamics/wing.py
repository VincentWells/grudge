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
from six.moves import zip




def make_wingmesh():
    import numpy
    from math import pi, cos, sin
    from meshpy.tet import MeshInfo, build
    from meshpy.geometry import GeometryBuilder, Marker, \
            generate_extrusion, make_box

    geob = GeometryBuilder()

    profile_marker = Marker.FIRST_USER_MARKER

    wing_length = 2
    wing_subdiv = 5

    rz_points = [
            (0, -wing_length*1.05),
            (0.7, -wing_length*1.05),
            ] + [
                (r, x) for x, r in zip(
                    numpy.linspace(-wing_length, 0, wing_subdiv, endpoint=False),
                    numpy.linspace(0.8, 1, wing_subdiv, endpoint=False))
            ] + [(1,0)] + [
                (r, x) for x, r in zip(
                    numpy.linspace(wing_length, 0, wing_subdiv, endpoint=False),
                    numpy.linspace(0.8, 1, wing_subdiv, endpoint=False))
            ][::-1] + [
            (0.7, wing_length*1.05),
            (0, wing_length*1.05)
            ]

    from meshpy.naca import get_naca_points
    geob.add_geometry(*generate_extrusion(
        rz_points=rz_points,
        base_shape=get_naca_points("0012", number_of_points=20),
        ring_markers=(wing_subdiv*2+4)*[profile_marker]))

    def deform_wing(p):
        x, y, z = p
        return numpy.array([
            x + 0.8*abs(z/wing_length)** 1.2,
            y + 0.1*abs(z/wing_length)**2,
            z])

    geob.apply_transform(deform_wing)

    points, facets, facet_markers = make_box(
            numpy.array([-1.5,-1,-wing_length-1], dtype=numpy.float64),
            numpy.array([3,1,wing_length+1], dtype=numpy.float64))

    geob.add_geometry(points, facets, facet_markers=facet_markers)

    mesh_info = MeshInfo()
    geob.set(mesh_info)
    mesh_info.set_holes([(0.5,0,0)])

    mesh = build(mesh_info)
    print("%d elements" % len(mesh.elements))

    fvi2fm = mesh.face_vertex_indices_to_face_marker

    face_marker_to_tag = {
            profile_marker: "noslip",
            Marker.MINUS_X: "inflow",
            Marker.PLUS_X: "outflow",
            Marker.MINUS_Y: "inflow",
            Marker.PLUS_Y: "inflow",
            Marker.PLUS_Z: "inflow",
            Marker.MINUS_Z: "inflow"
            }

    def bdry_tagger(fvi, el, fn, all_v):
        face_marker = fvi2fm[fvi]
        return [face_marker_to_tag[face_marker]]

    from grudge.mesh import make_conformal_mesh
    return make_conformal_mesh(mesh.points, mesh.elements, bdry_tagger)




def main():
    from grudge.backends import guess_run_context
    rcon = guess_run_context( ["cuda", "mpi"])

    if rcon.is_head_rank:
        mesh = make_wingmesh()
        #from grudge.mesh import make_rect_mesh
        #mesh = make_rect_mesh(
        #       boundary_tagger=lambda fvi, el, fn, all_v: ["inflow"])
        mesh_data = rcon.distribute_mesh(mesh)
    else:
        mesh_data = rcon.receive_mesh()

    for order in [3]:
        from pytools import add_python_path_relative_to_script
        add_python_path_relative_to_script("..")

        from gas_dynamics_initials import UniformMachFlow
        wing = UniformMachFlow(angle_of_attack=0)

        from grudge.models.gas_dynamics import GasDynamicsOperator
        op = GasDynamicsOperator(dimensions=3,
                gamma=wing.gamma, mu=wing.mu,
                prandtl=wing.prandtl, spec_gas_const=wing.spec_gas_const,
                bc_inflow=wing, bc_outflow=wing, bc_noslip=wing,
                inflow_tag="inflow", outflow_tag="outflow", noslip_tag="noslip")

        discr = rcon.make_discretization(mesh_data, order=order,
                        debug=["cuda_no_plan",
                            #"cuda_dump_kernels",
                            #"dump_dataflow_graph",
                            #"dump_optemplate_stages",
                            #"dump_dataflow_graph",
                            #"print_op_code"
                            "cuda_no_metis",
                            ],
                        default_scalar_type=numpy.float64,
                        tune_for=op.sym_operator())

        from grudge.visualization import SiloVisualizer, VtkVisualizer
        #vis = VtkVisualizer(discr, rcon, "shearflow-%d" % order)
        vis = SiloVisualizer(discr, rcon)

        fields = wing.volume_interpolant(0, discr)

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

        from grudge.timestep import RK4TimeStepper
        stepper = RK4TimeStepper()

        # diagnostics setup ---------------------------------------------------
        from logpyle import LogManager, add_general_quantities, \
                add_simulation_quantities, add_run_info

        logmgr = LogManager("navierstokes-%d.dat" % order, "w", rcon.communicator)
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
                    final_time=200,
                    #max_steps=500,
                    logmgr=logmgr,
                    max_dt_getter=lambda t: 0.6 * op.estimate_timestep(discr,
                        stepper=stepper, t=t, max_eigenvalue=max_eigval[0]))

            for step, t, dt in step_it:
                if step % 200 == 0:
                #if False:
                    visf = vis.make_file("wing-%d-%06d" % (order, step))

                    #rhs_fields = rhs(t, fields)

                    from pyvisfile.silo import DB_VARTYPE_VECTOR
                    from grudge.discretization import ones_on_boundary
                    vis.add_data(visf,
                            [
                                ("rho", discr.convert_volume(op.rho(fields), kind="numpy")),
                                ("e", discr.convert_volume(op.e(fields), kind="numpy")),
                                ("rho_u", discr.convert_volume(op.rho_u(fields), kind="numpy")),
                                ("u", discr.convert_volume(op.u(fields), kind="numpy")),

                                #("rhs_rho", discr.convert_volume(op.rho(rhs_fields), kind="numpy")),
                                #("rhs_e", discr.convert_volume(op.e(rhs_fields), kind="numpy")),
                                #("rhs_rho_u", discr.convert_volume(op.rho_u(rhs_fields), kind="numpy")),
                                ],
                            expressions=[
                                ("p", "(0.4)*(e- 0.5*(rho_u*u))"),
                                ],
                            time=t, step=step
                            )
                    visf.close()

                fields = stepper(fields, t, dt, rhs)
                t += dt

        finally:
            vis.close()
            logmgr.save()
            discr.close()




if __name__ == "__main__":
    main()

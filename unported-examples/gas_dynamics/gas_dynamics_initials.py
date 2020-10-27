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
import numpy
import numpy.linalg as la
from six.moves import range




class UniformMachFlow:
    def __init__(self, mach=0.1, p=1, rho=1, reynolds=100,
            gamma=1.4, prandtl=0.72, char_length=1, spec_gas_const=287.1,
            angle_of_attack=None, direction=None, gaussian_pulse_at=None,
            pulse_magnitude=0.1):
        """
        :param direction: is a vector indicating the direction of the
          flow. Only one of angle_of_attack and direction may be
          specified. Only the direction, not the magnitude, of
          direction is taken into account.

        :param angle_of_attack: if not None, specifies the angle of
          the flow along the Y axis, where the flow is
          directed along the X axis.
        """
        if angle_of_attack is not None and direction is not None:
            raise ValueError("Only one of angle_of_attack and "
                    "direction may be specified.")

        if angle_of_attack is None and direction is None:
            angle_of_attack = 0

        if direction is not None:
            self.direction = direction/la.norm(direction)
        else:
            self.direction = None

        self.mach = mach
        self.p = p
        self.rho = rho

        self.gamma = gamma
        self.prandtl = prandtl
        self.reynolds = reynolds
        self.length = char_length
        self.spec_gas_const = spec_gas_const

        self.angle_of_attack = angle_of_attack

        self.gaussian_pulse_at = gaussian_pulse_at
        self.pulse_magnitude = pulse_magnitude

        self.c = (self.gamma * p / rho)**0.5
        u = self.velocity = mach * self.c
        self.e = p / (self.gamma - 1) + rho / 2 * u**2

        if numpy.isinf(self.reynolds):
            self.mu = 0
        else:
            self.mu = u * self.length * rho / self.reynolds

    def direction_vector(self, dimensions):
        # this must be done here because dimensions is not known above
        if self.direction is None:
            assert self.angle_of_attack is not None
            direction = numpy.zeros(dimensions, dtype=numpy.float64)
            direction[0] = numpy.cos(
                    self.angle_of_attack / 180. * numpy.pi)
            direction[1] = numpy.sin(
                    self.angle_of_attack / 180. * numpy.pi)
            return direction
        else:
            return self.direction

    def __call__(self, t, x_vec):
        ones = numpy.ones_like(x_vec[0])
        rho_field = ones*self.rho

        if self.gaussian_pulse_at is not None:
            rel_to_pulse = [x_vec[i] - self.gaussian_pulse_at[i]
                    for i in range(len(x_vec))]
            rho_field +=  self.pulse_magnitude * self.rho * numpy.exp(
                - sum(rtp_i**2 for rtp_i in rel_to_pulse)/2)

        direction = self.direction_vector(x_vec.shape[0])

        from grudge.tools import make_obj_array
        u_field = make_obj_array([ones*self.velocity*dir_i
            for dir_i in direction])

        from grudge.tools import join_fields
        return join_fields(rho_field, self.e*ones, self.rho*u_field)

    def volume_interpolant(self, t, discr):
        return discr.convert_volume(
                        self(t, discr.nodes.T),
                        kind=discr.compute_kind,
                        dtype=discr.default_scalar_type)

    def boundary_interpolant(self, t, discr, tag):
        return discr.convert_boundary(
                        self(t, discr.get_boundary(tag).nodes.T),
                         tag=tag, kind=discr.compute_kind,
                         dtype=discr.default_scalar_type)

class Vortex:
    def __init__(self):
        self.beta = 5
        self.gamma = 1.4
        self.center = numpy.array([5, 0])
        self.velocity = numpy.array([1, 0])

        self.mu = 0
        self.prandtl = 0.72
        self.spec_gas_const = 287.1

    def __call__(self, t, x_vec):
        vortex_loc = self.center + t*self.velocity

        # coordinates relative to vortex center
        x_rel = x_vec[0] - vortex_loc[0]
        y_rel = x_vec[1] - vortex_loc[1]

        # Y.C. Zhou, G.W. Wei / Journal of Computational Physics 189 (2003) 159
        # also JSH/TW Nodal DG Methods, p. 209

        from math import pi
        r = numpy.sqrt(x_rel**2+y_rel**2)
        expterm = self.beta*numpy.exp(1-r**2)
        u = self.velocity[0] - expterm*y_rel/(2*pi)
        v = self.velocity[1] + expterm*x_rel/(2*pi)
        rho = (1-(self.gamma-1)/(16*self.gamma*pi**2)*expterm**2)**(1/(self.gamma-1))
        p = rho**self.gamma

        e = p/(self.gamma-1) + rho/2*(u**2+v**2)

        from grudge.tools import join_fields
        return join_fields(rho, e, rho*u, rho*v)

    def volume_interpolant(self, t, discr):
        return discr.convert_volume(
                        self(t, discr.nodes.T
                            .astype(discr.default_scalar_type)),
                        kind=discr.compute_kind)

    def boundary_interpolant(self, t, discr, tag):
        return discr.convert_boundary(
                        self(t, discr.get_boundary(tag).nodes.T
                            .astype(discr.default_scalar_type)),
                         tag=tag, kind=discr.compute_kind)







class Vortex:
    def __init__(self):
        self.beta = 5
        self.gamma = 1.4
        self.center = numpy.array([5, 0])
        self.velocity = numpy.array([1, 0])
        self.final_time = 0.5

        self.mu = 0
        self.prandtl = 0.72
        self.spec_gas_const = 287.1

    def __call__(self, t, x_vec):
        vortex_loc = self.center + t*self.velocity

        # coordinates relative to vortex center
        x_rel = x_vec[0] - vortex_loc[0]
        y_rel = x_vec[1] - vortex_loc[1]

        # Y.C. Zhou, G.W. Wei / Journal of Computational Physics 189 (2003) 159
        # also JSH/TW Nodal DG Methods, p. 209

        from math import pi
        r = numpy.sqrt(x_rel**2+y_rel**2)
        expterm = self.beta*numpy.exp(1-r**2)
        u = self.velocity[0] - expterm*y_rel/(2*pi)
        v = self.velocity[1] + expterm*x_rel/(2*pi)
        rho = (1-(self.gamma-1)/(16*self.gamma*pi**2)*expterm**2)**(1/(self.gamma-1))
        p = rho**self.gamma

        e = p/(self.gamma-1) + rho/2*(u**2+v**2)

        from grudge.tools import join_fields
        return join_fields(rho, e, rho*u, rho*v)

    def volume_interpolant(self, t, discr):
        return discr.convert_volume(
                        self(t, discr.nodes.T
                            .astype(discr.default_scalar_type)),
                        kind=discr.compute_kind)

    def boundary_interpolant(self, t, discr, tag):
        return discr.convert_boundary(
                        self(t, discr.get_boundary(tag).nodes.T
                            .astype(discr.default_scalar_type)),
                         tag=tag, kind=discr.compute_kind)





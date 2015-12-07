from . import base


class TrapezoidalIntegrator(base.Integrator):
    def integrate(self):
        return (self.h / 2) * (
            self.f(self.x(0)) +
            self.f(self.x(self.n)) +
            2 * sum((self.f(self.x(i)) for i in range(1, self.n))))

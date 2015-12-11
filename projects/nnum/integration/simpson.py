from . import base


class SimpsonIntegrator(base.Integrator):
    def integrate(self):
        return (self.h / 3) * (
            self.f(self.x(0)) +
            self.f(self.x(self.n)) +
            2 * sum((self.f(self.x(i)) for i in range(1, self.n))) +
            2 * sum((self.f(self.x(2 * i + 1)) for i in range(0, self.n // 2))))

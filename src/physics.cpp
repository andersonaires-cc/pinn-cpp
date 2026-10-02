#include "physics.hpp"


torch::Tensor Physics::residual(
    torch::Tensor u,
    torch::Tensor d2u_dt2
) {

    /*
     * Equação:
     *
     * u'' + u = 0
     *
     * Portanto:
     *
     * residual = u'' + u
     */

    return d2u_dt2 + u;
}
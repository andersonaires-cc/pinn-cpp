#include <iostream>
#include <iomanip>

#include <torch/torch.h>

#include "network.hpp"
#include "dataset.hpp"
#include "physics.hpp"

int main(){

    std::cout
        << "========================================\n";

    std::cout
        << "PINN - Oscilador Harmônico\n";

    std::cout
        << "========================================\n\n";

    PINN model(
        1,32,1
    );

    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(0.001)
    );


    PINNData dados =
        Dataset::gerar(100);

    const int epochs = 3000;


    for(
        int epoch = 1;
        epoch <= epochs;
        ++epoch
    ){
        
        optimizer.zero_grad();

        auto t =
            dados.t_fisica
                .clone()
                .detach();

        t.set_requires_grad(true);

        auto u =
            model->forward(t);


        auto du_dt =
            torch::autograd::grad(
                {u},
                {t},
                {torch::ones_like(u)},
                true,
                true
            )[0];


        auto d2u_dt2 =
            torch::autograd::grad(
                {du_dt},
                {t},
                {torch::ones_like(du_dt)},
                true,
                true
            )[0];

        auto residual =
            Physics::residual(
                u,
                d2u_dt2
            );
        
        auto loss_physics =
            torch::mean(
                torch::pow(
                    residual,
                    2
                )
            );

         // ========================================================
        // CONDIÇÃO INICIAL
        // ========================================================

        auto t0 =
            dados.t_inicial
                .clone()
                .detach();

        t0.set_requires_grad(true);


        auto u0 =
            model->forward(t0);


        auto loss_u0 =
            torch::mean(
                torch::pow(
                    u0 - 1.0f,
                    2
                )
            );


        auto du0_dt =
            torch::autograd::grad(
                {u0},
                {t0},
                {torch::ones_like(u0)},
                true,
                true
            )[0];


        auto loss_v0 =
            torch::mean(
                torch::pow(
                    du0_dt,
                    2
                )
            );


        auto loss =
            loss_physics
            + loss_u0
            + loss_v0;
        

        loss.backward();


        optimizer.step();


        if (
            epoch == 1 ||
            epoch % 100 == 0
        ) {

            std::cout
                << "Epoch: "
                << std::setw(4)
                << epoch

                << " | Loss: "
                << std::fixed
                << std::setprecision(8)
                << loss.item<float>()

                << " | Physics: "
                << loss_physics.item<float>()

                << " | IC u: "
                << loss_u0.item<float>()

                << " | IC v: "
                << loss_v0.item<float>()

                << "\n";
        }


    };

    std::cout
        << "\n========================================\n";

    std::cout
        << "Treinamento concluido!\n";

    std::cout
        << "========================================\n";


    return 0;

}
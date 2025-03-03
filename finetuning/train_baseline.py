import os
from pathlib import Path

from stable_baselines3 import PPO
    
from commonpower.control.configs.algorithms import SB3MetaConfig, SB3PPOConfig, SB3AlgorithmBaseConfig
from commonpower.control.logging_utils.loggers import *
from commonpower.control.runners import SingleAgentTrainer
from commonpower.control.wrappers import *
from scenarios import *
from utils import *


def run_experiment(
    save_path: str,
    sb3_config: SB3AlgorithmBaseConfig,
    forecast_horizon: timedelta,
    episode_length: int,
    train_sys: System,
    seed: int = 1,
    n_eps: int = 900,
    fixed_start: str = None,
    limited_date_range: List[datetime] = None
):
    train_config = SB3MetaConfig(
        total_steps=n_eps * episode_length,
        seed=seed,
        algorithm=PPO,
        algorithm_config=sb3_config,
    )

    # set up logger
    tb_log_dir = os.getcwd() + f'/tensorboard/{save_path}/{seed}'
    # logger = TensorboardLogger(log_dir=tb_log_dir, callback=SafetyCallback)
    # Uncomment to enable W&B logging --> have to change entity_name!
    logger = WandBLogger(
        log_dir=tb_log_dir,
        entity_name="srl4ps",
        project_name="commonpower_experiments",
        run_name=f"{save_path}_{seed}",
    )

    # specify the path where the model should be saved
    model_dir = os.getcwd() + f'/models/{save_path}/{seed}'

    wrappers = WrapperStack().add(SingleAgentWrapper)

    # start training
    runner = SingleAgentTrainer(
        sys=train_sys,
        wrapper=wrappers.get_stack(),
        alg_config=train_config,
        horizon=forecast_horizon,
        episode_length=episode_length,
        logger=logger,
        save_path=model_dir,
        seed=seed,
        continuous_control=True,
        limited_date_range=limited_date_range
    )
    runner.run(fixed_start=fixed_start)


if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5]
    n_eps = 900
    approach = Approach.WithProjectionSafeguard
    penalty = Penalty.DDPenalty
    scenario_constructor = Scenario.ConstantPricesScenario

    stage = Stage.Train
    forecast_length = 6
    forecast_cls = PersistenceForecaster(frequency=timedelta(hours=1), horizon=timedelta(hours=forecast_length), look_back=timedelta(hours=24)))

    scenario, deployment_runner = create_scenario(
        stage=stage, approach=approach, penalty=penalty, scenario_constructor=scenario_constructor.value
    )

    save_path = f'{scenario_constructor}/{approach}/{penalty}'

    # Optional: set start date for training (we work with data from 2016) and limit the date range for training data to a specific time
    train_start = "01.07.2016"
    date_format = "%Y-%m-%d %H:%M:00"
    start = datetime.strptime("2016-07-01 00:00:00", date_format)
    end = datetime.strptime("2016-07-31 23:00:00", date_format)

    # extract relevant parameters
    horizon = getattr(deployment_runner, "horizon")
    episode_length = 72

    # set up configuration for the PPO algorithm
    ppo_config = SB3PPOConfig(
        device="cpu",
        n_steps=4 * episode_length,
        batch_size=4 * episode_length,
        learning_rate=0.0008,
        n_epochs=5,
        policy_kwargs=dict(log_std_init=-2),
    )  # otherwise default hyperparameters for PPO

    for seed in seeds:
        run_experiment(
            save_path=save_path,
            seed=seed,
            n_eps=n_eps,
            sb3_config=ppo_config,
            forecast_horizon=horizon,
            episode_length=episode_length,
            train_sys=scenario,
            fixed_start=train_start,
            limited_date_range=[start, end]
        )

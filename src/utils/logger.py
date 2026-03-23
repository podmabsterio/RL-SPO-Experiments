from comet_ml import Experiment

def build_experiment(cfg, env_id) -> Experiment:
    experiment = Experiment(
        project_name=cfg.project_name,
    )
    run_name = cfg.get("run_name", None)
    if run_name == None:
        run_name = env_id
    else:
        run_name = f"{run_name}_env={env_id}"
    experiment.set_name(run_name)
    return experiment, run_name
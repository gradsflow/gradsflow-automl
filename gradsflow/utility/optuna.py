import optuna


def is_best_trial(study: optuna.Study, trial: optuna.Trial) -> bool:
    if study.best_trial.number == trial.number:
        return True
    return False

import optuna


class AutoModel:
    def __init__(self):
        self.study = optuna.create_study()

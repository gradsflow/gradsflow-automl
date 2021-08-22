from flash.text.classification import TextClassifier

from gradsflow.autoclassifier.base import AutoClassifier


# noinspection PyTypeChecker
class AutoTextClassifier(AutoClassifier):
    DEFAULT_BACKBONES = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "sgugger/tiny-distilbert-classification",
    ]

    def build_model(self, **kwargs):
        backbone = kwargs["backbone"]
        optimizer = kwargs["optimizer"]
        learning_rate = kwargs["lr"]

        return TextClassifier(
            self.num_classes,
            backbone=backbone,
            optimizer=self.OPTIMIZER_INDEX[optimizer],
            learning_rate=learning_rate,
        )

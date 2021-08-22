from flash.image.classification import ImageClassifier

from gradsflow.autoclassifier.base import AutoClassifier


# noinspection PyTypeChecker
class AutoImageClassifier(AutoClassifier):
    DEFAULT_BACKBONES = ["ssl_resnet18", "ssl_resnet50"]

    def build_model(self, **kwargs):
        backbone = kwargs["backbone"]
        optimizer = kwargs["optimizer"]
        learning_rate = kwargs["lr"]

        return ImageClassifier(
            self.num_classes,
            backbone=backbone,
            optimizer=self.OPTIMIZER_INDEX[optimizer],
            learning_rate=learning_rate,
        )

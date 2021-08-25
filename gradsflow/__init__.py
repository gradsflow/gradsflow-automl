"""An AutoML Library made with Optuna and PyTorch Lightning"""

from gradsflow.core.automodel import AutoModel
from gradsflow.tasks.autoclassification.image import AutoImageClassifier
from gradsflow.tasks.autoclassification.text import AutoTextClassifier
from gradsflow.tasks.summarization import AutoSummarization

__version__ = "0.0.1"

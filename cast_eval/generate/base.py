"""Abstract base class for all CAST evaluation generators."""
from abc import ABC, abstractmethod
from pathlib import Path


class TTSGenerator(ABC):
    """
    Interface for generating outputs (audio or text predictions) on the CAST benchmark.

    Audio generators write .wav files named <item_id>_00.wav (and optional .trans sidecars).
    Text generators write <item_id>.json files with predicted_stressed_words.

    evaluate_audio_metrics() from cast_eval.metrics handles both — it looks for .wav files
    for audio generators and .json files for text generators.
    """

    @abstractmethod
    def generate(self, dataset, output_dir: Path, **kwargs) -> None:
        """
        Run generation on the full dataset, writing outputs to output_dir.

        Args:
            dataset: HuggingFace Dataset with CAST schema (sentence, context, stressed_word, item_id, ...)
            output_dir: Directory to write output files
            **kwargs: Generator-specific parameters
        """
        ...

    def __repr__(self) -> str:
        return self.__class__.__name__

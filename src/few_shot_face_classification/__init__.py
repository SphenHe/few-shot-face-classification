"""Few-Shot Face Classification package."""
from few_shot_face_classification.cache import build_embeddings_cache
from few_shot_face_classification.main import add_none, detect_and_export, recognise, validate_labels
from few_shot_face_classification.people_count import count_people, count_people_in_folder

__all__ = [
    "add_none",
    "build_embeddings_cache",
    "count_people",
    "count_people_in_folder",
    "detect_and_export",
    "recognise",
    "validate_labels",
]

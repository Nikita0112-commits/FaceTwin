"""Custom exception hierarchy for the FaceTwin project."""


class FaceTwinError(Exception):
    """Base exception for all FaceTwin errors."""
    pass


class ConfigurationError(FaceTwinError):
    """Raised when configuration loading or validation fails."""
    pass


class VideoProcessingError(FaceTwinError):
    """Raised when video file loading, decoding, or preprocessing fails."""
    pass


class LandmarkExtractionError(FaceTwinError):
    """Raised when face detection or landmark extraction fails."""
    pass


class FeatureExtractionError(FaceTwinError):
    """Raised when computing facial movement or clinical features fails."""
    pass


class ModelTrainingError(FaceTwinError):
    """Raised during model training, convergence, or validation failures."""
    pass


class DigitalTwinStateError(FaceTwinError):
    """Raised when Kalman filter state estimation or progression tracking encounters invalid state."""
    pass


class ClinicalDataValidationError(FaceTwinError):
    """Raised when clinical metrics or UPDRS scores fail schema validation."""
    pass

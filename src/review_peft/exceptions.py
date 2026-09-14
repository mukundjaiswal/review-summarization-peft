"""Exception hierarchy."""


class ReviewPeftError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(ReviewPeftError):
    """Required configuration is missing or invalid."""


class DatasetError(ReviewPeftError):
    """A dataset is missing, malformed, or lacks a required column."""


class AdapterNotFoundError(ReviewPeftError):
    """The tuned variant was requested but no trained adapter exists."""

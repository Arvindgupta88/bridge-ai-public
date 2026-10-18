"""Fast, local application-readiness checks for the Streamlit UI."""

from dataclasses import dataclass

from src.core.config_validation import validate_configuration


@dataclass(frozen=True)
class ReadinessStatus:
    ready: bool
    configuration_valid: bool
    initialization_complete: bool


def check_readiness(
    *,
    provider_name: str,
    model: str,
    initialization_complete: bool,
) -> ReadinessStatus:
    """Validate selected configuration and startup completion without I/O."""
    configuration_valid = validate_configuration(
        provider_name=provider_name,
        model=model,
    ).is_valid
    return ReadinessStatus(
        ready=configuration_valid and initialization_complete,
        configuration_valid=configuration_valid,
        initialization_complete=initialization_complete,
    )

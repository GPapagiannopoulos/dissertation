"""Constants defined for the MOTOR port modules."""

from typing import Final

MOTOR_RMS_EPS: Final[float] = 1e-5

# Every block concatenates the normalised age and its square onto its input, so each
# block's projection is two columns wider than the model.
MOTOR_AGE_FEATURES: Final[int] = 2

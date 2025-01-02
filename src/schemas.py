from pedalboard._pedalboard import AudioProcessorParameter


class SynthParameter:
    def __init__(
        self,
        min_value: float,
        max_value: float,
        step_size: float,
        approximate_step_size: float,
        valid_values: list[float],
        type: type,
        **kwargs,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.step_size = step_size
        self.approximate_step_size = approximate_step_size
        self.valid_values = valid_values
        self.type = type

    def __str__(self):
        return f"SynthParameter(min_value={self.min_value}, max_value={self.max_value}, step_size={self.step_size}, approximate_step_size={self.approximate_step_size}, valid_values={self.valid_values}, type={self.type})"

    def __repr__(self):
        return f"SynthParameter(min_value={self.min_value}, max_value={self.max_value}, step_size={self.step_size}, approximate_step_size={self.approximate_step_size}, valid_values={self.valid_values}, type={self.type})"

    @classmethod
    def from_audio_processor_parameter(cls, parameter: AudioProcessorParameter):
        return cls(
            min_value=parameter.min_value,  # type: ignore
            max_value=parameter.max_value,  # type: ignore
            step_size=parameter.step_size,  # type: ignore
            approximate_step_size=parameter.approximate_step_size,  # type: ignore
            valid_values=parameter.valid_values,  # type: ignore
            type=parameter.type,
        )

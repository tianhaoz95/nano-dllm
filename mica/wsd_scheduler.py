from dataclasses import dataclass
from typing import List

import transformers


@dataclass
class WSDPhase:
    name: str
    block_size: int
    steps: int


class WSDBlockSizeScheduler:
    """
    Warmup–Stable–Decay block-size curriculum (LLaDA2.0 §4 / research plan §6.2).

    Maps optimizer step → block_size.  Phases are applied in order; once the
    cumulative step budget of a phase is exhausted, the next phase begins.
    """

    def __init__(self, phases: List[WSDPhase]):
        self.phases = phases
        # Precompute (start_step, phase) pairs for O(P) lookup
        cumulative = 0
        self._starts: List[tuple[int, WSDPhase]] = []
        for p in phases:
            self._starts.append((cumulative, p))
            cumulative += p.steps
        self.total_steps = cumulative

    # ── Core interface ─────────────────────────────────────────────────────

    def get_phase(self, step: int) -> WSDPhase:
        """Return the phase that owns optimizer `step`."""
        active = self.phases[-1]
        for start, phase in self._starts:
            if step >= start:
                active = phase
            else:
                break
        return active

    def get_block_size(self, step: int) -> int:
        return self.get_phase(step).block_size

    # ── Constructors ───────────────────────────────────────────────────────

    @classmethod
    def from_research_plan(cls, seq_len: int = 512) -> "WSDBlockSizeScheduler":
        """
        Default schedule from the MiCA-BD3LM research plan:
          warmup_ar(1,500) → warmup_4(4,500) → warmup_32(32,500) →
          warmup_128(128,500) → stable(seq_len,2000) →
          decay_64(64,300) → decay_32(32,200)
        Total: 4500 optimizer steps.
        """
        return cls([
            WSDPhase("warmup_ar",  block_size=1,       steps=500),
            WSDPhase("warmup_4",   block_size=4,       steps=500),
            WSDPhase("warmup_32",  block_size=32,      steps=500),
            WSDPhase("warmup_128", block_size=128,     steps=500),
            WSDPhase("stable",     block_size=seq_len, steps=2000),
            WSDPhase("decay_64",   block_size=64,      steps=300),
            WSDPhase("decay_32",   block_size=32,      steps=200),
        ])

    @classmethod
    def dry_run(cls) -> "WSDBlockSizeScheduler":
        """Tiny schedule for pipeline verification — finishes in seconds."""
        return cls([
            WSDPhase("warmup_ar", block_size=1,  steps=2),
            WSDPhase("warmup_4",  block_size=4,  steps=2),
            WSDPhase("stable",    block_size=64, steps=2),
            WSDPhase("decay_32",  block_size=32, steps=2),
        ])

    def __repr__(self) -> str:
        phases_str = ", ".join(
            f"{p.name}(bs={p.block_size}, steps={p.steps})"
            for p in self.phases
        )
        return f"WSDBlockSizeScheduler([{phases_str}], total={self.total_steps})"


class WSDBlockSizeCallback(transformers.TrainerCallback):
    """
    Updates trainer.block_size at the start of each data step according to
    the WSD schedule.  Prints a one-line notice whenever the phase changes.

    Note: on_step_begin receives state.global_step = number of *completed*
    optimizer steps, so block_size transitions happen on the exact step
    boundary defined in the schedule.
    """

    def __init__(self, trainer, scheduler: WSDBlockSizeScheduler):
        self._trainer = trainer
        self.scheduler = scheduler
        self._last_phase_name: str = ""

    def on_step_begin(self, args, state, control, **kwargs):
        step = state.global_step
        phase = self.scheduler.get_phase(step)
        new_size = phase.block_size

        # Always keep trainer in sync (cheap attr set)
        self._trainer.block_size = new_size

        # Log phase transitions once per phase
        if phase.name != self._last_phase_name and state.is_local_process_zero:
            print(
                f"\n  [WSD] step={step}  phase={phase.name}  "
                f"block_size={new_size}"
            )
            self._last_phase_name = phase.name

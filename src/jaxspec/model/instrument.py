from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from flax import nnx
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from ..data import ObsConfiguration


def redistribute(integrated_spectrum, old_e_low, old_e_high, e_low, e_high):
    # Suppose old_e_high[i] == old_e_low[i+1]
    edges = jnp.concatenate([old_e_low[:1], old_e_high])
    cumflux = jnp.concatenate([jnp.zeros(1), jnp.cumsum(integrated_spectrum)])
    return jnp.interp(e_high, edges, cumflux) - jnp.interp(e_low, edges, cumflux)


class GainModel(nnx.Module):
    """Generic gain model. ``__call__(energies)`` returns the per-energy gain factor."""

    @abstractmethod
    def __call__(self, energies: ArrayLike) -> ArrayLike: ...


class ConstantGain(GainModel):
    """A scalar gain factor, independent of energy.

    The factor lives as :attr:`factor` (an ``nnx.Param``). Its prior is provided
    via the unified prior dict under the key ``"instrument.gain.factor"``
    (shared across instrumented obs) or ``"instrument.gain.factor[*]"`` /
    ``"instrument.gain.factor[obs_name]"`` (per-obs).
    """

    def __init__(self):
        self.factor = nnx.Param(jnp.asarray(1.0))

    def __call__(self, energies: ArrayLike) -> ArrayLike:
        return self.factor[...]


class ShiftModel(nnx.Module):
    """Generic shift model. ``__call__(energies)`` returns shifted energies."""

    @abstractmethod
    def __call__(self, energies: ArrayLike) -> ArrayLike: ...


class ConstantShift(ShiftModel):
    """An additive energy shift, constant across the spectrum.

    The offset lives as :attr:`offset` (an ``nnx.Param``). Its prior is provided
    via the unified prior dict under the key ``"instrument.shift.offset"``
    (shared) or ``"instrument.shift.offset[*]"`` / ``"instrument.shift.offset[obs_name]"``
    (per-obs).
    """

    def __init__(self):
        self.offset = nnx.Param(jnp.asarray(0.0))

    def __call__(self, energies: ArrayLike) -> ArrayLike:
        return energies + self.offset[...]


class InstrumentModel(nnx.Module):
    """Per-observation instrument response.

    Pass as a dict to :class:`~jaxspec.fit.BayesianModel`::

        BayesianModel(
            spectral_model, prior, observations,
            instrument_model={
                "PN": None, # explicit reference
                "MOS1": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
                "MOS2": InstrumentModel(gain=ConstantGain(), shift=ConstantShift()),
            },
        )

    ``None`` entries (or simply omitting an observation) apply the identity
    fold (``transfer_matrix @ flux``) — useful for the reference instrument.

    Parameters:
        gain: Optional :class:`GainModel` (e.g. :class:`ConstantGain`). When
            ``None``, no flux scaling is applied.
        shift: Optional :class:`ShiftModel` (e.g. :class:`ConstantShift`). When
            ``None``, the input energies pass through unchanged.
    """

    #: When ``True``, :class:`~jaxspec.fit._forward_model.ForwardModel` builds
    #: the un-merged response components (``redistribution``, ``grouping``,
    #: ``area``, ``exposure``) into the per-observation cache passed to
    #: :meth:`fold`. Subclasses set this to ``True`` when their math needs the
    #: components separately (e.g. pileup, RMF calibration).
    requires_components = False

    def __init__(self, gain: GainModel | None = None, shift: ShiftModel | None = None):
        self.gain = gain
        self.shift = shift

    def default_prior(self, observation: ObsConfiguration, obs_name: str) -> dict:
        """Return data-dependent default priors scoped to this obs.

        Mirrors :meth:`~jaxspec.model.background.BackgroundModel.default_prior`:
        subclasses (e.g. pileup models with per-obs dead-time / grade-fraction
        parameters) override this to inject ``[obs_name]``-scoped defaults.
        User prior entries override these defaults.
        """
        return {}

    def apply_shift(self, energies: ArrayLike) -> ArrayLike:
        """Apply :attr:`shift` to ``energies`` and clip away non-positive values."""
        if self.shift is None:
            return energies
        return jnp.clip(self.shift(energies), min=1e-6)

    def apply_gain(self, flux, energies: ArrayLike):
        """Multiply ``flux`` (or each branch in a pytree) by :attr:`gain`'s factor."""
        if self.gain is None:
            return flux
        factor = jnp.clip(self.gain(energies), min=0.0)
        return jax.tree.map(lambda f: f * factor, flux)

    def fold(
        self,
        spectrum: ArrayLike,
        cache: dict,
        eval_energies: ArrayLike | None = None,
    ):
        """
        Fold the input spectrum (or branches of a pytree) into the instrument using the pre-computed transfer matrix.
        """

        # Current contract: shift is applied here if a grid is provided, else in
        # the forward model. The spectrum was integrated on the *unshifted*
        # ``eval_energies``, so the shift must go on the native grid: bin i
        # collects the flux from its shifted band, matching the no-grid path
        # (which evaluates flux_func on the shifted native energies directly).
        if eval_energies is not None:
            target_energies = self.apply_shift(cache["in_energies"])
            spectrum = jax.tree.map(
                lambda s: redistribute(s, *eval_energies, *target_energies), spectrum
            )

        spectrum = jax.tree.map(lambda s: self.apply_gain(s, cache["in_energies"]), spectrum)

        return jax.tree.map(lambda s: jnp.clip(cache["transfer_matrix"] @ s, min=1e-6), spectrum)
        
        
class PileupModel(InstrumentModel):
    """Per-observation instrument response with pileup effects included. Pileup formula from Davis 2001.

    Pass as a dict to :class:`~jaxspec.fit.BayesianModel`::

        BayesianModel(
            spectral_model, prior, observations,
            instrument_model={
                "Chandra": PileupModel(gain=ConstantGain(), shift=ConstantShift(), **pileup_kwargs),
            },
        )

    Parameters:
        gain and shift: Identical to the InstrumentModel()

        alpha: Grade migration factor : probability that the piled event is not rejected as "bad event"

        psf_frac: Fraction of events in the source extraction region to which pileup will be applied

        frame_time: frame time (of readout time) of the observation. In Chandra data it corresponds to the 'EXPTIME' keyword in the *_evt2.fits file

        frac_expo: good exposure time per frame. Between 0.0 and 1.0. In Chandra data it corresponds to the 'FRACEXPO' keyword

        g0: Optional. Grade correction for single photon detection, between 0.0 and 1.0. Default is 1.0

        npiled: Optional. Number of photons considered for pileup in a single frame. Default is 5

        num_regions: Optional. Number of regions to which pileup model will be applied independently. Default is 1.0, valid for point sources. Extended sources might require higher value.
    """

    requires_components = True

    def __init__(self, gain: GainModel | None = None, shift: ShiftModel | None = None, **kwargs):
        super().__init__(gain = gain, shift=shift)

        self.alpha = nnx.Param(jnp.asarray(0.5))
        self.psf_frac = nnx.Param(jnp.asarray(0.95))

        self.constants = {
            "frac_expo" : kwargs.get("frac_expo"),
            "frame_time" : kwargs.get("frame_time"),
            "num_regions" : kwargs.get("num_regions", 1.0),
            "g0" : kwargs.get("g0", 1.0),
            "npiled" : kwargs.get("npiled", 5.0),
        }

    def fold(
        self,
        spectrum: ArrayLike,
        cache: dict,
        eval_energies : ArrayLike | None=None,
    ):

        if eval_energies is None:
            raise ValueError("Eval energies cannot be None : an energy grid must be provided")

        # ARF convolution
        # cache["redistribution"] = to_jax_matrix(obs.redistribution.data, sparse=sparse)
        # cache["grouping"] = to_jax_matrix(obs.grouping.data, sparse=sparse)
        # cache["area"] = jnp.asarray(obs.area.data)
        # cache["exposure"] = jnp.asarray(obs.exposure.data)

        eval_energies = self.apply_shift(eval_energies)
        spectrum = jax.tree.map(lambda s: redistribute(s, *eval_energies, *cache["in_energies"]), spectrum)
        arf_s = jax.tree.map(lambda f: f * cache["area"], spectrum)

        num_regions = self.constants["num_regions"]
        fracexpo = self.constants["frac_expo"]
        frame_time = self.constants["frame_time"]
        g0 = self.constants["g0"]
        npiled = self.constants["npiled"]

        # Calculate pileup following original algorithm
        psf_frac = self.psf_frac / num_regions / fracexpo
        arf_s_tmp = arf_s * psf_frac
        integ_arf_s = jnp.sum(arf_s_tmp)

        results = arf_s * psf_frac  # term p=1 of the sum

        exp_factor = jnp.exp(-frame_time * integ_arf_s / g0)
        exp_factor *= num_regions * fracexpo

        # Normalize to avoid overflow and perform FFT convolutions
        arf_s_tmp /= integ_arf_s
        integ_arf_s_n = integ_arf_s  ## for renormalization after
        n_orig = arf_s_tmp.shape[-1]
        arf_s_fft = jnp.fft.rfft(arf_s_tmp)
        factor = 1

        # Offset required in case the energy grid does not start at zero
        ioff = - jnp.array(cache["in_energies"][0,0] // (cache["in_energies"][0,1] - cache["in_energies"][0,0]), jnp.int64)
        
        # Compute FFT with offset
        tmpar = jnp.array([arf_s_tmp[ie + ioff] for ie in range(n_orig)])
        arf_s_fft_2 = jnp.fft.rfft(tmpar)

        # Calculate higher order terms
        for i in range(2, npiled + 1):

            integ_arf_s_n *= integ_arf_s   #renormalization factor

            # Convolution via FFT
            arf_s_tmp = jnp.fft.irfft(arf_s_fft * arf_s_fft_2 ** (i-1), n=n_orig)
            arf_s_tmp = jnp.clip(arf_s_tmp, min=0)
            
            # Apply grade migration factor
            factor *= self.alpha * frame_time / i
            results += factor * integ_arf_s_n * arf_s_tmp 

        # Apply final corrections
        results *= exp_factor

        # Handle non-piled fraction
        remaining_frac = 1.0 - self.psf_frac
        results += arf_s * jnp.clip(remaining_frac, min=0)

        spectrum = self.apply_gain(results, cache["in_energies"])

        return jax.tree.map(lambda s: jnp.clip(cache["grouping"] @ cache["redistribution"] @ s * cache["exposure"] , min=1e-10), spectrum)

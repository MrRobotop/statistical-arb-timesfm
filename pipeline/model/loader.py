"""TimesFM model singleton with thread-safe lazy loading.

Uses the timesfm 1.3.0 API (timesfm.TimesFm / TimesFmHparams / TimesFmCheckpoint).
The CLAUDE.md documents the planned TimesFM 2.5 API (TimesFM_2p5_200M_torch /
ForecastConfig), which is not yet on PyPI. This module wraps the installed API
behind the same external interface so the forecaster layer is unaffected when
2.5 ships.

Quantile layout for default 200M checkpoint:
  The google/timesfm-1.0-200m-pytorch uses a fixed head of 1280.
  1280 / 128 (horizon) = 10 (mean + 9 quantiles).
  We use the default setup to avoid state_dict size mismatches.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# HuggingFace checkpoint for timesfm 1.3.0 (PyTorch)
_HF_REPO = "google/timesfm-1.0-200m-pytorch"

# Note: The 200M model checkpoint is hardcoded for a specific output head size.
# Using these exact quantiles and a horizon of 128 ensures the model loads correctly.
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QUANT_MEAN = 0
QUANT_Q10 = 1
QUANT_Q50 = 5
QUANT_Q90 = 9


class TimesFMLoader:
    """Singleton loader for the TimesFM model.

    Usage:
        loader = TimesFMLoader.get_instance()
        loader.load()                   # downloads weights once (~400 MB)
        predictions = loader.model.forecast(...)
    """

    _instance: TimesFMLoader | None = None
    _model = None
    _lock: threading.Lock = threading.Lock()

    # Stored for reference after load
    max_context: int = 512
    max_horizon: int = 128

    @classmethod
    def get_instance(cls) -> TimesFMLoader:
        """Return the singleton instance (thread-safe double-checked locking)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def load(self, max_context: int = 512, max_horizon: int = 128) -> None:
        """Download weights and initialise the model.

        Args:
            max_context: Input context length. Must be a multiple of 32 (patch
                size). Rounded up automatically.
            max_horizon: Maximum forecast horizon. For 200M checkpoint, use 128.

        Raises:
            RuntimeError: If RAM is too low or HuggingFace auth fails.
        """
        self._preflight_checks()

        # Enforce patch-size alignment
        if max_context % 32 != 0:
            max_context = ((max_context // 32) + 1) * 32
            logger.warning("max_context rounded up to %d (must be ∝32)", max_context)

        TimesFMLoader.max_context = max_context
        TimesFMLoader.max_horizon = max_horizon

        # Set precision before importing torch so it applies globally
        import torch
        torch.set_float32_matmul_precision("high")

        # Authenticate with HuggingFace if token is available
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            try:
                import huggingface_hub
                huggingface_hub.login(token=hf_token, add_to_git_credential=False)
                logger.info("HuggingFace login successful")
            except Exception as exc:
                logger.warning("HuggingFace login warning: %s", exc)

        import timesfm  # noqa: PLC0415 — intentional deferred import

        logger.info("Loading TimesFM from %s …", _HF_REPO)
        t0 = time.perf_counter()

        try:
            model = timesfm.TimesFm(
                hparams=timesfm.TimesFmHparams(
                    context_len=max_context,
                    horizon_len=max_horizon,
                    backend="cpu",
                    quantiles=QUANTILES,
                ),
                checkpoint=timesfm.TimesFmCheckpoint(
                    huggingface_repo_id=_HF_REPO,
                ),
            )
        except Exception as exc:
            # Provide a helpful error for the most common failure: missing auth
            if "401" in str(exc) or "Unauthorized" in str(exc) or "RepositoryNotFound" in str(exc):
                raise RuntimeError(
                    f"HuggingFace authentication failed for {_HF_REPO}.\n"
                    "  1. Add HF_TOKEN=<your-token> to your .env file.\n"
                    "  2. Accept model terms at https://huggingface.co/google/timesfm-1-0-200m-pytorch\n"
                    "  3. Get a token at https://huggingface.co/settings/tokens\n"
                    f"  Original error: {exc}"
                ) from exc
            raise

        TimesFMLoader._model = model
        elapsed = time.perf_counter() - t0
        logger.info("TimesFM ready in %.1f s", elapsed)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        """Return True if weights have been loaded."""
        return TimesFMLoader._model is not None

    @property
    def model(self):
        """Return the loaded model, raising if not yet loaded."""
        if TimesFMLoader._model is None:
            raise RuntimeError(
                "TimesFM model is not loaded. Call TimesFMLoader.get_instance().load() first."
            )
        return TimesFMLoader._model

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    @staticmethod
    def _preflight_checks() -> None:
        """Warn/fail on insufficient RAM or disk space."""
        try:
            import psutil
            mem = psutil.virtual_memory()
            avail_gb = mem.available / 1024 ** 3

            if avail_gb < 2.0:
                raise RuntimeError(
                    f"Insufficient RAM: {avail_gb:.1f} GB available (need ≥ 2 GB)"
                )
            if avail_gb < 4.0:
                logger.warning(
                    "Available RAM %.1f GB < 4 GB — model loading may be slow",
                    avail_gb,
                )

            disk = psutil.disk_usage(".")
            free_gb = disk.free / 1024 ** 3
            if free_gb < 2.0:
                logger.warning(
                    "Free disk space %.1f GB < 2 GB — model download may fail",
                    free_gb,
                )
        except ImportError:
            logger.warning("psutil not available; skipping RAM/disk checks")

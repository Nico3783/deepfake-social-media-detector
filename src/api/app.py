"""
FastAPI application factory for the deepfake detection service.

Purpose: Create and configure the FastAPI application with all middleware and routes.
Responsibilities: App creation, middleware setup, model loading, startup/shutdown events.
Dependencies: fastapi, src.api.routes, src.config

Research Traceability:
    Research Objective: Deployable deepfake detection service
    Methodology: RESTful API with proper lifecycle management
    Implementation: src/api/app.py
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import router, set_predictors
from src.config.settings import load_settings
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan: load model on startup, cleanup on shutdown.

    This function handles the model loading lifecycle. On startup it
    loads the configured model and initializes both image and video
    predictors. On shutdown it releases resources.
    """
    settings = load_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Startup: device=%s", device)

    try:
        # Lazy imports to avoid slow startup when model is not needed
        from src.models.model_factory import ModelFactory
        from src.inference.predict_image import ImagePredictor
        from src.inference.predict_video import VideoPredictor

        model_path = Path(settings.get("model_path", "outputs/checkpoints/best_model.pth"))

        if model_path.exists():
            model_type = settings.get("model_type", "xception")
            model = ModelFactory.create_model(model_type, num_classes=2)
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
            model.to(device)
            model.eval()
            logger.info("Loaded model: %s from %s", model_type, model_path)

            image_predictor = ImagePredictor(
                model=model,
                device=device,
                target_size=299 if model_type == "xception" else 224,
            )
            video_predictor = VideoPredictor(
                model=model,
                device=device,
                frame_sample_rate=settings.get("frame_sample_rate", 5),
                target_size=299 if model_type == "xception" else 224,
            )

            set_predictors(image_predictor, video_predictor, model_type, str(device))
        else:
            logger.warning("No model checkpoint found at %s — running without model", model_path)
            set_predictors(None, None, "none", str(device))

    except Exception as e:
        logger.error("Model loading failed: %s", e)
        set_predictors(None, None, "none", str(device))

    yield

    logger.info("Shutdown: releasing resources")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI instance.
    """
    app = FastAPI(
        title="Deepfake Detection API",
        description=(
            "RESTful API for detecting deepfake images and videos "
            "using deep learning (XceptionNet / EfficientNet)."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(router)

    return app

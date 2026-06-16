# TODO

## Phase 0: Project Foundation [IN PROGRESS]

- [x] Fix invalid `src/{__init__.py}` filename
- [x] Create `__init__.py` for all packages
- [x] Create `.gitignore`, `requirements.txt`, `pyproject.toml`
- [x] Implement `src/config/settings.py`, `paths.py`, `constants.py`
- [x] Implement `src/utils/logger.py`, `seed.py`, `helpers.py`, `file_manager.py`
- [x] Create all 5 config YAML files
- [x] Create `README.md`, `CHANGELOG.md`, `TODO.md`
- [ ] Initialize git repository
- [ ] Implement remaining `__init__.py` with proper imports

## Phase 1: Data Pipeline [PENDING]

- [ ] `src/data/download.py` - Dataset downloaders
- [ ] `src/data/organize.py` - Dataset organization
- [ ] `src/data/splitter.py` - Train/val/test splitting
- [ ] `src/data/dataset.py` - PyTorch Dataset class
- [ ] `src/data/metadata.py` - Metadata management

## Phase 2: Preprocessing Pipeline [PENDING]

- [ ] `src/preprocessing/face_detector.py` - MTCNN face detection
- [ ] `src/preprocessing/cropper.py` - Face cropping and alignment
- [ ] `src/preprocessing/transforms.py` - Image transformations
- [ ] `src/preprocessing/frame_extractor.py` - Video frame extraction

## Phase 3: Model Architecture [PENDING]

- [ ] `src/models/xception.py` - XceptionNet implementation
- [ ] `src/models/efficientnet.py` - EfficientNet implementation
- [ ] `src/models/factory.py` - Model factory

## Phase 4: Training Pipeline [PENDING]

- [ ] `src/training/trainer.py` - Training loop
- [ ] `src/training/losses.py` - Loss functions
- [ ] `src/training/callbacks.py` - Training callbacks
- [ ] `src/training/metrics.py` - Metric calculations

## Phase 5: Evaluation [PENDING]

- [ ] `src/evaluation/evaluate.py` - Evaluation pipeline
- [ ] `src/evaluation/confusion_matrix.py` - Confusion matrix
- [ ] `src/evaluation/roc_auc.py` - ROC-AUC analysis
- [ ] `src/evaluation/report.py` - Report generation

## Phase 6: Video Detection [PENDING]

- [ ] `src/inference/aggregate.py` - Frame-to-video aggregation
- [ ] `src/inference/confidence.py` - Confidence scoring

## Phase 7: Visualization [PENDING]

- [ ] `src/visualization/training_curves.py` - Training plots
- [ ] `src/visualization/roc_curve.py` - ROC curve
- [ ] `src/visualization/confusion_matrix.py` - Confusion matrix
- [ ] `src/visualization/grad_cam.py` - GradCAM visualization

## Phase 8: API [PENDING]

- [ ] `src/api/app.py` - FastAPI application
- [ ] `src/api/routes.py` - API endpoints
- [ ] `src/api/models.py` - Request/response models

## Phase 9: Testing [PENDING]

- [ ] Unit tests for all modules
- [ ] Integration tests
- [ ] Test configuration

## Phase 10: Documentation [PENDING]

- [ ] API documentation
- [ ] Thesis asset generation
- [ ] Final cleanup

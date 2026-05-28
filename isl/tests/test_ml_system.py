"""
Tests for the ML-based ISL recognition system.
Validates landmark extraction, model loading, inference, and engine integration.
"""

import os
import sys
import json
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─── Feature Extraction Tests ─────────────────────────────────────────────

class TestFeatureExtraction:
    """Test the landmark normalization and feature extraction."""
    
    def test_normalize_no_hands(self):
        from src.ml.extract_landmarks import _normalize_landmarks, FEATURE_DIM
        features = _normalize_landmarks(None, None, None)
        assert features.shape == (FEATURE_DIM,)
        assert features[-2] == 0.0  # no right hand
        assert features[-1] == 0.0  # no left hand
    
    def test_normalize_right_hand_only(self):
        from src.ml.extract_landmarks import _normalize_landmarks, FEATURE_DIM
        rh = np.random.randn(21, 3).astype(np.float32) * 0.1 + 0.5
        features = _normalize_landmarks(rh, None, None)
        assert features.shape == (FEATURE_DIM,)
        assert features[-2] == 1.0  # right hand present
        assert features[-1] == 0.0  # no left hand
        # Right hand features should be non-zero
        assert np.any(features[:63] != 0)
        # Left hand features should be zero
        assert np.all(features[63:126] == 0)
    
    def test_normalize_both_hands(self):
        from src.ml.extract_landmarks import _normalize_landmarks, FEATURE_DIM
        rh = np.random.randn(21, 3).astype(np.float32) * 0.1 + 0.5
        lh = np.random.randn(21, 3).astype(np.float32) * 0.1 + 0.5
        features = _normalize_landmarks(rh, lh, None)
        assert features[-2] == 1.0
        assert features[-1] == 1.0
    
    def test_normalize_with_pose(self):
        from src.ml.extract_landmarks import _normalize_landmarks, FEATURE_DIM
        rh = np.random.randn(21, 3).astype(np.float32) * 0.1 + 0.5
        pose = np.random.randn(33, 4).astype(np.float32) * 0.1 + 0.5
        # Make shoulders have realistic separation
        pose[11, :3] = [0.6, 0.4, 0.0]  # left shoulder
        pose[12, :3] = [0.4, 0.4, 0.0]  # right shoulder
        features = _normalize_landmarks(rh, None, pose)
        assert features.shape == (FEATURE_DIM,)
        # Pose features should be non-zero
        assert np.any(features[126:156] != 0)
    
    def test_feature_dim_constant(self):
        from src.ml.extract_landmarks import FEATURE_DIM
        # 21*3*2 (hands) + 10*3 (pose) + 2 (flags) = 158
        assert FEATURE_DIM == 158


# ─── Dataset Tests ─────────────────────────────────────────────────────────

class TestDataset:
    """Test dataset loading and augmentation."""
    
    @pytest.fixture
    def data_dir(self):
        return "extracted_data"
    
    @pytest.mark.skipif(
        not os.path.exists("extracted_data/metadata.json"),
        reason="Extracted data not available"
    )
    def test_metadata_exists(self, data_dir):
        meta_path = os.path.join(data_dir, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        assert meta["num_classes"] == 71
        assert meta["successful"] == 1120
        assert meta["failed"] == 0
        assert meta["feature_dim"] == 158
    
    @pytest.mark.skipif(
        not os.path.exists("extracted_data/metadata.json"),
        reason="Extracted data not available"
    )
    def test_dataset_loading(self, data_dir):
        from src.ml.dataset import ISLDataset
        dataset = ISLDataset(data_dir, seq_length=30, augment=False)
        assert len(dataset) == 1120
        assert dataset.num_classes == 71
        
        # Get a sample
        seq, label = dataset[0]
        assert seq.shape == (30, 158)
        assert isinstance(label, torch.Tensor)
        assert 0 <= label.item() < 71
    
    @pytest.mark.skipif(
        not os.path.exists("extracted_data/metadata.json"),
        reason="Extracted data not available"
    )
    def test_augmentation(self, data_dir):
        from src.ml.dataset import ISLDataset
        dataset = ISLDataset(data_dir, seq_length=30, augment=True)
        seq1, label1 = dataset[0]
        seq2, label2 = dataset[0]
        # Same sample but augmented differently
        assert label1 == label2
        # Sequences should differ due to augmentation
        assert not torch.allclose(seq1, seq2)


# ─── Model Tests ───────────────────────────────────────────────────────────

class TestModel:
    """Test model architecture and forward pass."""
    
    def test_full_model_forward(self):
        from src.ml.model import ISLModel
        model = ISLModel(input_dim=158, hidden_dim=128, num_layers=2, num_classes=71)
        x = torch.randn(4, 30, 158)
        logits = model(x)
        assert logits.shape == (4, 71)

    def test_hybrid_model_forward(self):
        from src.ml.model import ISLModelHybrid
        model = ISLModelHybrid(input_dim=158, hidden_dim=128, num_layers=2, num_classes=71)
        x = torch.randn(4, 30, 158)
        lengths = torch.LongTensor([30, 24, 18, 12])
        logits = model(x, lengths=lengths)
        assert logits.shape == (4, 71)

    def test_transformer_model_forward(self):
        from src.ml.model import ISLModelTransformer
        model = ISLModelTransformer(input_dim=158, hidden_dim=128, num_layers=2, num_classes=71)
        x = torch.randn(4, 30, 158)
        lengths = torch.LongTensor([30, 24, 18, 12])
        logits = model(x, lengths=lengths)
        assert logits.shape == (4, 71)

    def test_tcn_model_forward(self):
        from src.ml.model import ISLModelTCN
        model = ISLModelTCN(input_dim=158, hidden_dim=128, num_layers=3, num_classes=71)
        x = torch.randn(4, 30, 158)
        lengths = torch.LongTensor([30, 24, 18, 12])
        logits = model(x, lengths=lengths)
        assert logits.shape == (4, 71)
    
    def test_lite_model_forward(self):
        from src.ml.model import ISLModelLite
        model = ISLModelLite(input_dim=158, hidden_dim=64, num_classes=71)
        x = torch.randn(4, 30, 158)
        logits = model(x)
        assert logits.shape == (4, 71)
    
    def test_attention_mechanism(self):
        from src.ml.model import TemporalAttention
        attn = TemporalAttention(hidden_dim=128)
        x = torch.randn(2, 30, 128)
        output = attn(x)
        assert output.shape == (2, 128)
    
    def test_attention_with_mask(self):
        from src.ml.model import TemporalAttention
        attn = TemporalAttention(hidden_dim=128)
        x = torch.randn(2, 30, 128)
        mask = torch.zeros(2, 30, dtype=torch.bool)
        mask[:, 20:] = True  # Mask last 10 frames
        output = attn(x, mask)
        assert output.shape == (2, 128)


# ─── Recognizer Tests ─────────────────────────────────────────────────────

class TestRecognizer:
    """Test the ML recognizer."""
    
    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pt"),
        reason="Trained model not available"
    )
    def test_model_loading(self):
        from src.ml.recognizer import MLRecognizer
        rec = MLRecognizer("models/best_model.pt")
        assert rec.num_classes == 71
        assert len(rec.class_names) == 71
        assert rec._ml_available if hasattr(rec, '_ml_available') else True
    
    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pt"),
        reason="Trained model not available"
    )
    def test_prediction_with_no_frames(self):
        from src.ml.recognizer import MLRecognizer
        rec = MLRecognizer("models/best_model.pt")
        sign, conf = rec.predict()
        assert sign is None
        assert conf == 0.0
    
    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pt"),
        reason="Trained model not available"
    )
    def test_top_k_predictions(self):
        from src.ml.recognizer import MLRecognizer
        rec = MLRecognizer("models/best_model.pt")
        
        # Feed some frames with hand data
        for _ in range(15):
            rh = np.random.randn(21, 3).astype(np.float32) * 0.1 + 0.5
            pose = np.random.randn(33, 4).astype(np.float32) * 0.1 + 0.5
            pose[11, :3] = [0.6, 0.4, 0.0]
            pose[12, :3] = [0.4, 0.4, 0.0]
            rec.add_frame(rh, None, pose)
        
        top = rec.get_top_k(5)
        assert len(top) == 5
        for name, prob in top:
            assert isinstance(name, str)
            assert 0 <= prob <= 1
    
    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pt"),
        reason="Trained model not available"
    )
    def test_reset(self):
        from src.ml.recognizer import MLRecognizer
        rec = MLRecognizer("models/best_model.pt")
        for _ in range(10):
            rec.add_frame(np.random.randn(21, 3).astype(np.float32), None, None)
        rec.reset()
        assert len(rec.frame_buffer) == 0
        assert rec._last_prediction is None


# ─── ML Engine Tests ──────────────────────────────────────────────────────

class TestMLEngine:
    """Test the ML-based sign engine."""
    
    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pt"),
        reason="Trained model not available"
    )
    def test_engine_initialization(self):
        from src.recognition.ml_engine import MLSignEngine
        engine = MLSignEngine(enable_voice=False)
        assert engine._ml_available
        assert engine.recognizer is not None
    
    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pt"),
        reason="Trained model not available"
    )
    def test_engine_process_frame(self):
        from src.recognition.ml_engine import MLSignEngine, SignResult
        from src.core.hand_tracker import TrackingResult
        
        engine = MLSignEngine(enable_voice=False)
        
        tracking = TrackingResult(
            right_hand=np.random.randn(21, 3).astype(np.float32) * 0.1 + 0.5,
            pose=np.random.randn(33, 4).astype(np.float32) * 0.1 + 0.5,
            frame_width=640,
            frame_height=480,
        )
        
        result = engine.process_frame(tracking)
        assert isinstance(result, SignResult)
    
    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pt"),
        reason="Trained model not available"
    )
    def test_engine_sentence_building(self):
        from src.recognition.ml_engine import MLSignEngine
        engine = MLSignEngine(enable_voice=False)
        
        engine._commit_sign("Hello")
        assert engine.get_sentence() == "Hello"
        
        # Wait for cooldown
        engine.last_committed_time = 0  # Reset cooldown
        engine._commit_sign("Morning")
        assert engine.get_sentence() == "Hello Morning"
        
        engine.backspace()
        assert engine.get_sentence() == "Hello"
        
        engine.clear_sentence()
        assert engine.get_sentence() == ""
    
    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pt"),
        reason="Trained model not available"
    )
    def test_engine_no_hands(self):
        from src.recognition.ml_engine import MLSignEngine, SignResult
        from src.core.hand_tracker import TrackingResult
        
        engine = MLSignEngine(enable_voice=False)
        tracking = TrackingResult(frame_width=640, frame_height=480)
        result = engine.process_frame(tracking)
        assert result.sign is None


# ─── Integration Tests ─────────────────────────────────────────────────────

class TestIntegration:
    """Test full pipeline integration."""
    
    def test_pipeline_import(self):
        """Pipeline should import ML engine when model exists."""
        from src.pipeline.fast_pipeline import SignEngine
        assert SignEngine.__name__ in ("MLSignEngine", "SignEngine")
    
    @pytest.mark.skipif(
        not os.path.exists("models/best_model.pt"),
        reason="Trained model not available"
    )
    def test_pipeline_uses_ml(self):
        from src.pipeline.fast_pipeline import SignEngine
        assert SignEngine.__name__ == "MLSignEngine"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

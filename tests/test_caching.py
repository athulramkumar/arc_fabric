"""
Tests for the simple caching strategy across Wan2.1 and LTX.

Tests three levels:
1. Unit tests for caching logic (no GPU required)
2. Integration test for Wan2.1 caching (requires GPU + weights)
3. Integration test for LTX caching (requires GPU + weights)
"""

import os
import sys
import time
import unittest

# ---------------------------------------------------------------------------
# 1. UNIT TESTS — pure-Python logic, no GPU needed
# ---------------------------------------------------------------------------

# Import the Wan2.1 caching utility directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'wan21'))

from api.utils.caching import should_use_cache, get_cache_statistics, validate_cache_config


class TestShouldUseCache(unittest.TestCase):
    """Verify step-skipping logic matches the notebook specification."""

    def test_before_start_always_fresh(self):
        for step in range(10):
            self.assertFalse(
                should_use_cache(step, start_step=10, end_step=40, interval=3),
                f"Step {step} should be fresh (before start)")

    def test_start_step_is_fresh(self):
        self.assertFalse(
            should_use_cache(10, start_step=10, end_step=40, interval=3))

    def test_cache_zone_pattern(self):
        """Interval=3 → fresh on 0,3,6,... offsets; cache otherwise."""
        results = []
        for step in range(10, 20):
            results.append(should_use_cache(step, 10, 40, 3))
        # offset:  0     1     2     3     4     5     6     7     8     9
        expected = [False, True, True, False, True, True, False, True, True, False]
        self.assertEqual(results, expected)

    def test_after_end_always_fresh(self):
        for step in range(41, 50):
            self.assertFalse(
                should_use_cache(step, start_step=10, end_step=40, interval=3),
                f"Step {step} should be fresh (after end)")

    def test_end_step_boundary(self):
        # step 40: offset=30, 30%3==0 → fresh (interval boundary)
        self.assertFalse(
            should_use_cache(40, start_step=10, end_step=40, interval=3))
        # step 39: offset=29, 29%3==2 → cached
        self.assertTrue(
            should_use_cache(39, start_step=10, end_step=40, interval=3))

    def test_none_end_step_caches_until_end(self):
        self.assertTrue(
            should_use_cache(99, start_step=10, end_step=None, interval=3))

    def test_interval_2(self):
        """Interval=2 → every other step is cached."""
        results = [should_use_cache(s, 0, None, 2) for s in range(6)]
        expected = [False, True, False, True, False, True]
        self.assertEqual(results, expected)


class TestCacheStatistics(unittest.TestCase):

    def test_no_caching_zone(self):
        stats = get_cache_statistics(total_steps=50, start_step=50, end_step=50, interval=3)
        self.assertEqual(stats["cache_hits"], 0)

    def test_full_run(self):
        stats = get_cache_statistics(50, start_step=10, end_step=40, interval=3)
        self.assertGreater(stats["cache_hits"], 0)
        self.assertGreater(stats["fresh_computes"], 0)
        self.assertEqual(stats["cache_hits"] + stats["fresh_computes"], 50)
        self.assertGreater(stats["estimated_speedup"], 1.0)

    def test_interval_3_hit_rate(self):
        """With interval=3, ~2/3 of steps in the zone should be cached."""
        stats = get_cache_statistics(50, start_step=0, end_step=None, interval=3)
        self.assertAlmostEqual(stats["cache_hit_rate"], 2 / 3, delta=0.05)


class TestValidateConfig(unittest.TestCase):

    def test_valid_config(self):
        ok, msg = validate_cache_config(50, 10, 40, 3)
        self.assertTrue(ok, msg)

    def test_start_too_large(self):
        ok, _ = validate_cache_config(50, 60, None, 3)
        self.assertFalse(ok)

    def test_interval_1_rejected(self):
        ok, _ = validate_cache_config(50, 0, None, 1)
        self.assertFalse(ok)

    def test_end_before_start(self):
        ok, _ = validate_cache_config(50, 20, 10, 3)
        self.assertFalse(ok)


class TestWanStaticMethod(unittest.TestCase):
    """Test the _should_use_cache static method added to WanT2V.
    Requires torch (run with af-wan21 env)."""

    def test_matches_utility(self):
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'wan21'))
            from wan.text2video import WanT2V
        except ImportError:
            self.skipTest("torch not available - run with af-wan21 env")
        for step in range(50):
            expected = should_use_cache(step, 10, 40, 3)
            actual = WanT2V._should_use_cache(step, 10, 40, 3)
            self.assertEqual(actual, expected, f"Mismatch at step {step}")


# ---------------------------------------------------------------------------
# 2. GPU INTEGRATION TEST — Wan2.1
# ---------------------------------------------------------------------------

class TestWan21CachingIntegration(unittest.TestCase):
    """End-to-end test: generate a short video with and without caching,
    verify caching is faster and output is valid."""

    WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'weights', 'wan21')
    CKPT_1_3B = os.path.join(WEIGHTS_DIR, 'Wan2.1-T2V-1.3B')

    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No GPU available")
        if not os.path.isdir(cls.CKPT_1_3B):
            raise unittest.SkipTest(f"Weights not found at {cls.CKPT_1_3B}")

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'wan21'))
        from wan.text2video import WanT2V
        from wan.configs.wan_t2v_1_3B import t2v_1_3B

        cls.pipe = WanT2V(
            config=t2v_1_3B,
            checkpoint_dir=cls.CKPT_1_3B,
            device_id=0, rank=0,
            t5_fsdp=False, dit_fsdp=False, use_usp=False, t5_cpu=False,
        )

    def _generate(self, **cache_kwargs):
        return self.pipe.generate(
            input_prompt="a cat sitting on a windowsill",
            size=(832, 480),
            frame_num=17,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=20,
            guide_scale=5.0,
            seed=42,
            offload_model=False,
            **cache_kwargs,
        )

    def test_baseline_produces_video(self):
        video = self._generate()
        self.assertIsNotNone(video)
        self.assertIn(video.dim(), (3, 4))  # C, N, H, W
        print(f"Baseline video shape: {video.shape}")

    def test_cached_produces_video(self):
        video = self._generate(
            cache_start_step=3, cache_end_step=None, cache_interval=3)
        self.assertIsNotNone(video)
        print(f"Cached video shape: {video.shape}")

    def test_caching_is_faster(self):
        # Warm up
        self._generate()

        t0 = time.time()
        self._generate()
        baseline_time = time.time() - t0

        t0 = time.time()
        self._generate(cache_start_step=3, cache_end_step=None, cache_interval=3)
        cached_time = time.time() - t0

        speedup = baseline_time / cached_time
        print(f"Baseline: {baseline_time:.2f}s, Cached: {cached_time:.2f}s, Speedup: {speedup:.2f}x")
        self.assertGreater(speedup, 1.05, "Caching should provide measurable speedup")

    def test_same_seed_different_cache_configs(self):
        """Different cache configs with same seed should produce videos of same shape."""
        import torch
        v1 = self._generate()
        v2 = self._generate(cache_start_step=5, cache_end_step=15, cache_interval=2)
        self.assertEqual(v1.shape, v2.shape)


# ---------------------------------------------------------------------------
# 3. GPU INTEGRATION TEST — LTX-Video
# ---------------------------------------------------------------------------

class TestLTXCachingIntegration(unittest.TestCase):
    """End-to-end test for LTX caching."""

    WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'weights', 'ltx_video', 'ltxv-2b-0.9.8-distilled')
    CKPT = os.path.join(WEIGHTS_DIR, 'ltxv-2b-0.9.8-distilled.safetensors')

    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("No GPU available")
        if not os.path.isfile(cls.CKPT):
            raise unittest.SkipTest(f"Weights not found at {cls.CKPT}")

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'ltx_video'))
        from ltx_video.inference import create_ltx_video_pipeline, seed_everething

        cls.pipeline = create_ltx_video_pipeline(
            ckpt_path=cls.CKPT,
            precision="bfloat16",
            text_encoder_model_name_or_path=cls.WEIGHTS_DIR,
            device="cuda",
        )

    def _generate(self, num_steps=8, cache_start_step=None, cache_end_step=None, cache_interval=3):
        import torch
        from ltx_video.inference import seed_everething
        from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy

        seed_everething(42)
        generator = torch.Generator(device="cuda").manual_seed(42)

        height, width, num_frames = 256, 256, 9
        height_padded = ((height - 1) // 32 + 1) * 32
        width_padded = ((width - 1) // 32 + 1) * 32
        num_frames_padded = ((num_frames - 2) // 8 + 1) * 8 + 1

        result = self.pipeline(
            num_inference_steps=num_steps,
            guidance_scale=3.0,
            skip_layer_strategy=SkipLayerStrategy.AttentionValues,
            generator=generator,
            output_type="pt",
            callback_on_step_end=None,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=24,
            prompt="a cat sitting on a windowsill",
            prompt_attention_mask=None,
            negative_prompt="worst quality",
            negative_prompt_attention_mask=None,
            media_items=None,
            conditioning_items=None,
            is_video=True,
            vae_per_channel_normalize=True,
            image_cond_noise_scale=0.0,
            mixed_precision=False,
            offload_to_cpu=False,
            device="cuda",
            enhance_prompt=False,
            cache_start_step=cache_start_step,
            cache_end_step=cache_end_step,
            cache_interval=cache_interval,
        ).images
        return result

    def test_baseline_produces_video(self):
        images = self._generate()
        self.assertIsNotNone(images)
        self.assertEqual(images.dim(), 5)  # B, C, F, H, W
        print(f"LTX baseline shape: {images.shape}")

    def test_cached_produces_video(self):
        images = self._generate(
            cache_start_step=2, cache_end_step=None, cache_interval=2)
        self.assertIsNotNone(images)
        print(f"LTX cached shape: {images.shape}")

    def test_caching_is_faster(self):
        # Warm up
        self._generate()

        t0 = time.time()
        self._generate()
        baseline_time = time.time() - t0

        t0 = time.time()
        self._generate(cache_start_step=2, cache_end_step=None, cache_interval=2)
        cached_time = time.time() - t0

        speedup = baseline_time / cached_time
        print(f"LTX Baseline: {baseline_time:.2f}s, Cached: {cached_time:.2f}s, Speedup: {speedup:.2f}x")
        self.assertGreater(speedup, 1.05, "Caching should provide measurable speedup")


if __name__ == '__main__':
    unittest.main(verbosity=2)

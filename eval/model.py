"""
Model loading and KV cache capture for text-only and VLM models.
"""

import torch


# Diverse paragraphs for prompt extension
_PARAGRAPHS = [
    "The theory of general relativity, published by Albert Einstein in 1915, "
    "describes gravity not as a force but as a curvature of spacetime caused by "
    "mass and energy. This revolutionary framework replaced Newton's law of "
    "universal gravitation for extreme conditions such as strong gravitational "
    "fields and high velocities approaching the speed of light.",

    "One of its most striking predictions is the existence of black holes, regions "
    "of spacetime where gravity is so intense that nothing, not even light, can "
    "escape. Einstein's field equations relate the geometry of spacetime to the "
    "distribution of matter within it.",

    "The solutions to these equations have led to profound discoveries including "
    "gravitational waves, confirmed by LIGO in 2015, and the expanding universe, "
    "first observed by Edwin Hubble. General relativity remains one of the two "
    "pillars of modern physics alongside quantum mechanics.",

    "Quantum mechanics governs the behavior of particles at the smallest scales. "
    "The wave function describes the probability amplitude for finding a particle "
    "in a given state. The Heisenberg uncertainty principle places fundamental "
    "limits on the precision with which certain pairs of physical properties can "
    "be simultaneously known.",

    "The Standard Model of particle physics classifies all known elementary particles "
    "and describes three of the four fundamental forces: electromagnetic, weak, and "
    "strong interactions. The Higgs boson, discovered at CERN in 2012, confirms the "
    "mechanism by which particles acquire mass.",

    "Thermodynamics and statistical mechanics provide the bridge between microscopic "
    "physics and macroscopic observations. The second law of thermodynamics states "
    "that the total entropy of an isolated system can only increase over time. This "
    "arrow of time is one of the deep unsolved problems in fundamental physics.",

    "The cosmic microwave background radiation, discovered in 1965 by Penzias and "
    "Wilson, provides a snapshot of the universe approximately 380,000 years after "
    "the Big Bang. Its tiny temperature fluctuations encode information about the "
    "density perturbations that seeded the formation of galaxies and large-scale "
    "structure we observe today.",

    "Dark matter and dark energy together constitute approximately 95 percent of the "
    "total energy content of the universe. Despite decades of experimental searches, "
    "the particle nature of dark matter remains unknown. Dark energy, responsible for "
    "the accelerating expansion of the universe, is even more mysterious.",

    "String theory proposes that the fundamental constituents of nature are not "
    "point-like particles but one-dimensional strings vibrating at different "
    "frequencies. The theory requires extra spatial dimensions beyond the three we "
    "observe, typically compactified at scales far below current experimental reach.",

    "The information paradox arises from the apparent conflict between quantum "
    "mechanics and general relativity at the event horizon of a black hole. Hawking "
    "radiation suggests that black holes eventually evaporate, raising the question "
    "of what happens to the information encoded in the matter that fell in.",
]


def generate_synthetic_kv(n_layers: int = 32, n_heads: int = 8, n_tokens: int = 256,
                           d: int = 128, device: str = "cuda") -> dict:
    """Generate synthetic KV cache tensors with realistic statistics."""
    print(f"Generating synthetic KV cache: {n_layers}L x {n_heads}H x {n_tokens}T x {d}d")
    torch.manual_seed(42)

    keys = torch.randn(n_layers, n_heads, n_tokens, d, device=device)
    values = torch.randn(n_layers, n_heads, n_tokens, d, device=device)

    layer_scales = torch.linspace(0.5, 2.0, n_layers, device=device).view(-1, 1, 1, 1)
    pos_scales = torch.linspace(0.8, 1.2, n_tokens, device=device).view(1, 1, -1, 1)
    keys = keys * layer_scales * pos_scales
    values = values * layer_scales * pos_scales

    return {
        "keys": keys, "values": values,
        "n_layers": n_layers, "n_heads": n_heads,
        "n_tokens": n_tokens, "head_dim": d,
    }


def load_model_and_capture_kv(model_name: str = "Qwen/Qwen2.5-3B-Instruct",
                               prompt: str = None,
                               device: str = "cuda",
                               min_tokens: int = 0,
                               image_path: str = None):
    """
    Load a causal LM (or VLM), run a forward pass, and capture the KV cache tensors.

    Returns:
        kv_cache, model, tokenizer, input_ids
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    is_vlm = image_path is not None

    print(f"Loading model {model_name}{'  (VLM mode)' if is_vlm else ''}...")

    if is_vlm:
        from transformers import (Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor,
                                  Qwen2VLImageProcessor, Qwen2VLVideoProcessor)
        ip = Qwen2VLImageProcessor.from_pretrained(model_name)
        vp = Qwen2VLVideoProcessor.from_pretrained(model_name)
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        processor = Qwen2_5_VLProcessor(image_processor=ip, tokenizer=tok, video_processor=vp)
        tokenizer = processor.tokenizer
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, dtype=torch.float16, device_map=device, trust_remote_code=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=device, trust_remote_code=True,
        )
    model.eval()

    if prompt is None:
        prompt = " ".join(_PARAGRAPHS[:3])

    # Extend prompt if min_tokens is requested
    if min_tokens > 0:
        test_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        current_tokens = test_ids.shape[1]
        para_idx = 0
        while current_tokens < min_tokens:
            prompt += " " + _PARAGRAPHS[para_idx % len(_PARAGRAPHS)]
            para_idx += 1
            test_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
            current_tokens = test_ids.shape[1]
        test_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        if test_ids.shape[1] > min_tokens:
            test_ids = test_ids[:, :min_tokens]
            prompt = tokenizer.decode(test_ids[0], skip_special_tokens=True)

    if is_vlm:
        from qwen_vl_utils import process_vision_info

        print(f"Loading image: {image_path}")
        vlm_prompt = prompt if prompt else "Describe this image in detail."

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": vlm_prompt},
            ]}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(device)
        input_ids = inputs["input_ids"]
        n_tokens = input_ids.shape[1]
        print(f"Running VLM forward pass ({n_tokens} tokens: image + text)...")

        with torch.no_grad():
            outputs = model(**inputs, use_cache=True)
            past_kv = outputs.past_key_values
    else:
        print(f"Running forward pass with {len(prompt)} char prompt...")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        n_tokens = input_ids.shape[1]
        print(f"  Tokens: {n_tokens}")

        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            past_kv = outputs.past_key_values

    # Extract KV cache into tensors
    from transformers.cache_utils import DynamicCache
    if isinstance(past_kv, DynamicCache):
        n_layers = len(past_kv.layers)
        sample_k = past_kv.layers[0].keys
    else:
        n_layers = len(past_kv)
        sample_k = past_kv[0][0]

    n_kv_heads = sample_k.shape[1]
    head_dim = sample_k.shape[3]

    print(f"  KV cache: {n_layers} layers, {n_kv_heads} KV heads, head_dim={head_dim}")

    keys = torch.zeros(n_layers, n_kv_heads, n_tokens, head_dim, dtype=torch.float32, device=device)
    values = torch.zeros_like(keys)

    for layer_idx in range(n_layers):
        if isinstance(past_kv, DynamicCache):
            keys[layer_idx] = past_kv.layers[layer_idx].keys[0].float()
            values[layer_idx] = past_kv.layers[layer_idx].values[0].float()
        else:
            keys[layer_idx] = past_kv[layer_idx][0][0].float()
            values[layer_idx] = past_kv[layer_idx][1][0].float()

    return {
        "keys": keys, "values": values,
        "n_layers": n_layers, "n_heads": n_kv_heads,
        "n_tokens": n_tokens, "head_dim": head_dim,
    }, model, tokenizer, input_ids

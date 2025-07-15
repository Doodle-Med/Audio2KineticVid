import torch
from transformers import AutoTokenizer
# Use AutoGPTQ for loading GPTQ model if available, else fall back to AutoModel
try:
    from auto_gptq import AutoGPTQForCausalLM
except ImportError:
    AutoGPTQForCausalLM = None
from transformers import AutoModelForCausalLM

# Cache models and tokenizers
_llm_cache = {}  # {model_name: (model, tokenizer)}

def list_available_llm_models():
    """Return a list of available LLM models for prompt generation"""
    return [
        "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ", 
        "microsoft/phi-2",
        "TheBloke/Llama-2-7B-Chat-GPTQ",
        "TheBloke/zephyr-7B-beta-GPTQ",
        "stabilityai/stablelm-2-1_6b"
    ]

def _load_llm(model_name):
    """Load LLM model and tokenizer, with caching"""
    global _llm_cache
    if model_name not in _llm_cache:
        print(f"Loading LLM model: {model_name}...")
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        
        # Load model (prefer AutoGPTQ if available for quantized model)
        if "GPTQ" in model_name and AutoGPTQForCausalLM:
            model = AutoGPTQForCausalLM.from_quantized(
                model_name, 
                use_safetensors=True,
                device="cuda", 
                use_triton=False,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto", 
                torch_dtype=torch.float16, 
                trust_remote_code=True
            )
            
        # Ensure model in eval mode
        model.eval()
        _llm_cache[model_name] = (model, tokenizer)
    
    return _llm_cache[model_name]

def generate_scene_prompts(
    segments, 
    llm_model="TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
    prompt_template=None,
    style_suffix="cinematic, 35 mm, shallow depth of field, film grain",
    max_tokens=100
):
    """
    Generate a visual scene description prompt for each lyric segment.
    
    Args:
        segments: List of segment dictionaries with 'text' field containing lyrics
        llm_model: Name of the LLM model to use
        prompt_template: Custom prompt template with {lyrics} placeholder
        style_suffix: Style keywords to append to scene descriptions
        max_tokens: Maximum new tokens to generate
        
    Returns:
        List of prompt strings corresponding to the segments
    """
    # Use default prompt template if none provided
    if not prompt_template:
        prompt_template = (
            "You are a cinematographer generating a scene for a music video. "
            "Describe one vivid visual scene (one sentence) that matches the mood and imagery of these lyrics, "
            "focusing on setting, atmosphere, lighting, and framing. Do not mention the artist or singing. "
            "Lyrics: \"{lyrics}\"\nScene description:"
        )
    
    model, tokenizer = _load_llm(llm_model)
    scene_prompts = []
    
    for seg in segments:
        lyrics = seg["text"]
        # Format prompt template with lyrics
        if "{lyrics}" in prompt_template:
            instruction = prompt_template.format(lyrics=lyrics)
        else:
            # Fallback if template doesn't have {lyrics} placeholder
            instruction = f"{prompt_template}\n\nLyrics: \"{lyrics}\"\nScene description:"
            
        # Encode input and generate
        inputs = tokenizer(instruction, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens, 
                temperature=0.7, 
                do_sample=True, 
                top_p=0.9, 
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Process generated text
        generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        
        # Ensure we got a sentence; if model returned multiple sentences, take first.
        if "." in generated:
            generated = generated.split(".")[0].strip() + "."
            
        # Append style suffix for Stable Diffusion
        prompt = generated
        if style_suffix and style_suffix.strip() and style_suffix not in prompt.lower():
            prompt = f"{prompt.strip()}, {style_suffix}"
            
        scene_prompts.append(prompt)
        
    return scene_prompts
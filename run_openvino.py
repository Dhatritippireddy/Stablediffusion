from optimum.intel.openvino.modeling_diffusion import OVStableDiffusionPipeline
 
pipeline = OVStableDiffusionPipeline.from_pretrained(
    "rupeshs/LCM-dreamshaper-v7-openvino",
    ov_config={"CACHE_DIR": ""},
)
prompt = "RED SHOES WITH FLORAL DESIGN "
 
images = pipeline(
    prompt=prompt,
    width=512,
    height=512,
    num_inference_steps=4,
    guidance_scale=1.0,
).images
images[0].save("out_image.png")
import torch

# Check if CUDA (GPU) is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Print the device where the model is loaded
print(f'Model loaded on: {device}')

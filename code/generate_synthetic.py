from diffusers import DDPMPipeline
import numpy as np

model_id = "google/ddpm-cifar10-32"

ddpm = DDPMPipeline.from_pretrained(model_id)
ddpm.to("cuda")

for i in range(49):
    images = ddpm(batch_size=1_000, output_type="np").images
    np.save(f"generated/generate-cifar10-{i}", images)

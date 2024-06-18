from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from diffusers import DiffusionPipeline
import torch
import os

app = FastAPI()

# Ensure the generated_images directory exists
os.makedirs("generated_images", exist_ok=True)

# Load the model
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", 
                                         torch_dtype=torch.float16, 
                                         use_safetensors=True, 
                                         variant="fp16")
pipe = pipe.to("cuda")  # Make sure to use "cuda" or "cpu" based on your setup

@app.get("/", response_class=HTMLResponse)
def form():
    return """
    <html>
        <body>
            <form action="/generate-image/" method="post">
                <input type="text" name="prompt" />
                <input type="submit" />
            </form>
        </body>
    </html>
    """

@app.post("/generate-image/", response_class=HTMLResponse)
async def generate_image(prompt: str = Form(...)):
    try:
        # Generate an image
        images = pipe(prompt=prompt).images[0]
        
        # Save the generated image
        file_path = f"generated_images/{prompt.replace(' ', '_')}.png"
        images.save(file_path)

        # Return HTML content displaying the image
        return f"""
        <html>
            <body>
                <h1>Generated Image for Prompt: {prompt}</h1>
                <img src="/generated_images/{prompt.replace(' ', '_')}.png" alt="Generated Image">
            </body>
        </html>
        """
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Static Files
app.mount("/generated_images", StaticFiles(directory="generated_images"), name="generated_images")

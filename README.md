# Understanding-Diffusion-Models-as-ResNets

# 🧠 Diffusion Models as ResNets

**Author:** Mahesh Godavarti  
**Tagline:** Deep Learning / AI Enthusiast  
**GitHub:** [mahesh-godavarti](https://github.com/mahesh-godavarti)  
**LinkedIn:** [linkedin.com/in/maheshgodavarti](https://www.linkedin.com/in/maheshgodavarti/)

---

## 📘 Overview

This project explores how **Diffusion Models** — the foundation of modern generative AI — can be understood as **deep Residual Networks (ResNets)** performing step-by-step denoising.

The accompanying slide deck and Python scripts provide an intuitive, educational walkthrough of the connection between diffusion processes and residual learning.

---

## 🧩 Repository Contents

| File | Description |
|------|--------------|
| `Diffusion_Models_as_ResNets_Redesigned.pptx` | Presentation explaining diffusion models as deep ResNets |
| `mnist_toy_resnet_diffusion.py` | Toy MNIST example illustrating the diffusion–ResNet analogy |
| `generate_step_grids_from_checkpoint.py` | Creates visual grids showing intermediate denoising steps |
| `generate_movie_from_images.py` | Combines saved images into an animation of the diffusion process |

---

## 🚀 Key Ideas

- Diffusion models transform random noise into coherent images through **iterative denoising**.  
- Each denoising step can be seen as a **residual update** — making diffusion models equivalent to very deep ResNets.  
- The formulation unifies several well-known generative approaches:
  - **DDPM / DDIM:** Noise prediction  
  - **Flow Matching:** Velocity field learning  

> 🧩 *Takeaway:* Diffusion = Residual Learning Across Time

---

## 🧪 How to Run the Code

1. **Install dependencies**
   ```bash
   pip install torch torchvision matplotlib numpy

# Image Classification & Object Detection

This project uses Stable Diffusion for image-to-image generation, PyTorch for training, and Hugging Face models to fine-tune and generate images.

Workflow of this project:
1. Load and process input images from a dataset. (Dataset was collected from Kaggle)
2. Fine-tune a pre-trained model using the data.
3. Generate transformed images or outputs based on the input images.

Project Structure:
1. dataset_prep.py: This script defines a dataset class for image-to-image translation tasks, learning mappings from input images to target images.
2. model.py: This script loads a Stable Diffusion model pipeline, using a pretrained model and transferring it to the specified device.
3. requirements.txt: Lists all required dependencies to run the project.

N.B: Could not complete the full project due to limited knowledge and Time Constraint. 

# Screw Defect Detection
This project aims to detect defects in screws by leveraging deep learning models for segmentation and classification. The workflow involves fine-tuning the Mask2Former model for precise segmentation of defective regions in screw images and using the Qwen2-VL model for few-shot classification to identify the defect type.

## Dataset:
The dataset folder contains two folders:
1. segemented_images: Contains the images for fine-tuning training the Mask2Former model.
2. valid_images: Contains the images for inferencing the model and for the Final pipeline.

## Mask2Former Finetuning:
This folder contains the code and configurations for fine-tuning the Mask2Former model from scratch. The model is trained to segment defective areas in screw images based on the provided segmentation masks.

#### Training Procedure:
- The model is fine-tuned on the segmented_images folder.
- The training pipeline adjusts hyperparameters like learning rate, batch size, and optimizer settings to achieve optimal performance for segmentation tasks.
- The final model is saved and used for inference on the validation images.

## outputs3:
This folder contains the following:
1. inference_results_image: Batch inference results for the valid_images after fine-tuning the Mask2Former model.
2. valid_preds: Validation images produced at each epoch during training to monitor model performance.
3. model_test_infer: The final output after inferencing the model using the pipeline, where the screw defects are segmented and classified.

## Final.ipynb:
This Jupyter notebook contains the complete pipeline:
1. Segmentation: The Mask2Former model is used to detect and highlight defective regions in screw images.
2. Classification: The Qwen2-VL model classifies the defect types using few-shot learning, providing a textual classification of the defect (e.g., "Scratch Neck," "Thread Top," etc.) based on the segmented images.

## Final Fine-Tuned Model
The fine-tuned Mask2Former model for screw defect segmentation is available for download and further use at: [Mask2Former Screw Defect Segmentation on Hugging Face](https://huggingface.co/shrish23/Mask2Former-screwdefect-segmentation/tree/main)

## Hardware:
The project was trained and tested using an **NVIDIA A100 80GB GPU**, which provided the necessary computational power for efficient training and inference, particularly for large models like Mask2Former and Qwen2-VL.

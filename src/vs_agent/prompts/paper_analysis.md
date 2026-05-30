You are analyzing a research paper for a personal method-card library.

The user studies virtual staining and biomedical image translation, but the paper does NOT need to explicitly mention virtual staining to be useful. General image-to-image translation, medical image generation, image restoration, domain adaptation, registration, super-resolution, diffusion, GAN, transformer, and training/evaluation papers can all be relevant.

Return strict JSON only:

{
  "paper_type": "",
  "main_task": "",
  "input_modality": "",
  "output_modality": "",
  "data_setting": "",
  "core_methods": [],
  "losses": [],
  "evaluation_metrics": [],
  "relevance_to_user_project": "high|medium|low|unclear",
  "relevance_reason": "",
  "useful_aspects": []
}

Rules:
- First identify the paper type and task.
- Summarize input/output and data setting, such as paired, unpaired, registered, unregistered, synthetic, clinical, microscopy, pathology, RGB, H&E, fluorescence, etc.
- Extract the core technical ideas, not keywords.
- Explain transferable value for virtual staining or biomedical image conversion.
- If the paper is pix2pix or another general image-to-image translation paper, classify it as general image-to-image translation and explain transferable value.
- Use only the supplied paper digest.


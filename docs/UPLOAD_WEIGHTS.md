# Publishing the weights (maintainer note)

One Hugging Face model repository holds everything `run.sh` needs besides this git repo:

```
Qwen3.8-Next-SDnvfp2/
  model-*.safetensors (207 shards, ~98 GB)   the 2-bit-expert checkpoint (expert tensors 4 codes/byte, same names as NVFP4)
  model.safetensors.index.json
  tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, chat_template.jinja, generation_config.json
  config.json, hf_quant_config.json          (the checkpoint's own; run.sh overlays checkpoint_config/ from this repo)
  dense_codes/                               1.8 GB of <module>.pt NVFP4 codes for the dense layers (loaded via SGLANG_DENSE_NVFP4_CODES_DIR)
```

Upload from the box that has them:

```bash
pip install -U huggingface_hub
hf auth login
hf upload-large-folder sdworld/Qwen3.8-Next-SDnvfp2 /path/to/checkpoint --repo-type model      # resumable, parallel
hf upload sdworld/Qwen3.8-Next-SDnvfp2 /path/to/dense_codes dense_codes --repo-type model
```

Then replace `sdworld` in `README.md`. Do **not** upload the `.complete.json` conversion markers,
`conversion_environment.json`, `qualification-notes.md`, `smoke_report.json`,
`validate_*_report.json`, `gsm8k_metrics.json`, `aime26_metrics.json`, `audit_unchanged_report.json`
that sit next to the shards on the build box — they describe the conversion. Add a model card
(`README.md` in the HF repo) with the numbers table from this repo and the base model's license.

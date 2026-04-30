# #!/usr/bin/env python3
# """
# merge_checkpoint.py  —  Patch lora_alpha bug and merge LoRA into base model

# Usage:
#   python merge_checkpoint.py --step 35
#   python merge_checkpoint.py --step 80
#   python merge_checkpoint.py --step 120 --lora-alpha 32 --lora-rank 16
#   python merge_checkpoint.py --step 35 --exp baseline  # for non-exp1 runs

# Output:
#   checkpoints/<exp>/global_step_<N>/actor/huggingface_merged/
# """
# import json
# import argparse
# import torch
# from peft import PeftModel
# from transformers import AutoModelForCausalLM, AutoTokenizer


# def merge(step, exp, lora_alpha, lora_rank):
#     base      = f"checkpoints/{exp}/global_step_{step}/actor/huggingface"
#     adapter   = f"{base}/lora_adapter"
#     output    = f"checkpoints/{exp}/global_step_{step}/actor/huggingface_merged"

#     # ── Patch lora_alpha (VERL bug: always saves 0) ───────────────────────────
#     cfg_path = f"{adapter}/adapter_config.json"
#     with open(cfg_path) as f:
#         cfg = json.load(f)

#     print(f"adapter_config before patch: lora_alpha={cfg.get('lora_alpha')}  r={cfg.get('r')}")
#     cfg["lora_alpha"] = lora_alpha
#     cfg["r"]          = lora_rank
#     with open(cfg_path, "w") as f:
#         json.dump(cfg, f, indent=2)
#     print(f"Patched: lora_alpha={lora_alpha}  r={lora_rank}")

#     # ── Load base + adapter, merge, save ─────────────────────────────────────
#     print(f"Loading base model from {base} ...")
#     model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16)
#     print(f"Loading LoRA adapter from {adapter} ...")
#     model = PeftModel.from_pretrained(model, adapter)
#     print("Merging LoRA into base weights ...")
#     model = model.merge_and_unload()
#     model.save_pretrained(output)

#     tokenizer = AutoTokenizer.from_pretrained(base)
#     tokenizer.save_pretrained(output)

#     print(f"\n✓ Merged checkpoint saved to: {output}")
#     print(f"\nEval commands:")
#     print(f"  python exp1_evaluate_upd2.py     --checkpoint {output} --split test")
#     print(f"  python exp1_evaluate_feedback.py --checkpoint {output} --split test")


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--step",       type=int,   required=True,  help="Global step number")
#     ap.add_argument("--exp",        type=str,   default="exp1", help="Experiment folder name (default: exp1)")
#     ap.add_argument("--lora-alpha", type=int,   default=32,     help="lora_alpha used in training (default: 32)")
#     ap.add_argument("--lora-rank",  type=int,   default=16,     help="lora_rank used in training (default: 16)")
#     args = ap.parse_args()

#     merge(args.step, args.exp, args.lora_alpha, args.lora_rank)


#!/usr/bin/env python3
"""
merge_checkpoint.py  —  Merge VERL LoRA checkpoint into base model

Usage:
  python merge_checkpoint.py --step 140
  python merge_checkpoint.py --step 200
  python merge_checkpoint.py --step 140 --exp exp1_full
  python merge_checkpoint.py --step 140 --upload --hf-repo username/model-name

Output:
  checkpoints/<exp>/global_step_<N>/actor/huggingface_merged_step<N>/
"""
import argparse
import torch
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"


def merge(step, exp, lora_rank, lora_alpha, upload, hf_repo, private):
    pt_path  = f"checkpoints/{exp}/global_step_{step}/actor/model_world_size_1_rank_0.pt"
    hf_path  = f"checkpoints/{exp}/global_step_{step}/actor/huggingface"
    output   = f"checkpoints/{exp}/global_step_{step}/actor/huggingface_merged_step{step}"

    print(f"Step       : {step}")
    print(f"Experiment : {exp}")
    print(f"PT path    : {pt_path}")
    print(f"Output     : {output}\n")

    print(f"Loading base model: {BASE_MODEL_ID} ...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float16)

    print(f"Wrapping with LoRA (r={lora_rank}, alpha={lora_alpha}) ...")
    lora_config = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules="all-linear", lora_dropout=0.0)
    model = get_peft_model(model, lora_config)

    print(f"Loading state dict from {pt_path} ...")
    sd = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(sd)

    print("Merging LoRA into base weights ...")
    model = model.merge_and_unload()

    print(f"Saving to {output} ...")
    model.save_pretrained(output)
    tokenizer = AutoTokenizer.from_pretrained(hf_path)
    tokenizer.save_pretrained(output)

    print(f"\n✓ Merged checkpoint saved to: {output}")
    print(f"\nEval commands:")
    print(f"  python exp1_evaluate_upd2.py     --checkpoint {output} --split test")
    print(f"  python exp1_evaluate_feedback.py --checkpoint {output} --split test --no-judge")

    if upload:
        if not hf_repo:
            print("\nERROR: --hf-repo is required when --upload is set")
            return
        from huggingface_hub import HfApi
        api = HfApi()
        print(f"\nCreating HF repo: {hf_repo} (private={private}) ...")
        try:
            api.create_repo(repo_id=hf_repo, repo_type="model", private=private)
        except Exception as e:
            print(f"  Repo may already exist: {e}")
        print(f"Uploading to {hf_repo} ...")
        api.upload_folder(folder_path=output, repo_id=hf_repo, repo_type="model")
        print(f"✓ Uploaded to https://huggingface.co/{hf_repo}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--step",       type=int,   required=True,       help="Global step number")
    ap.add_argument("--exp",        type=str,   default="exp1_full", help="Experiment folder (default: exp1_full)")
    ap.add_argument("--lora-rank",  type=int,   default=16,          help="LoRA rank (default: 16)")
    ap.add_argument("--lora-alpha", type=int,   default=32,          help="LoRA alpha (default: 32)")
    ap.add_argument("--upload",     action="store_true",             help="Upload to HuggingFace after merging")
    ap.add_argument("--hf-repo",    type=str,   default="",          help="HF repo e.g. username/model-name")
    ap.add_argument("--public",     action="store_true",             help="Make HF repo public (default: private)")
    args = ap.parse_args()

    merge(args.step, args.exp, args.lora_rank, args.lora_alpha, args.upload, args.hf_repo, not args.public)
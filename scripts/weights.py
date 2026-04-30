import transformers, logging, numpy as np, json, torch
from pathlib import Path
logging.basicConfig(level=logging.INFO)

model_name = "gpt2-medium" # gpt-2, gpt2-medium, gpt2-large, gpt2-xl
current_dir = Path.cwd()
output_dir = current_dir / "weights" / model_name
output_dir.mkdir(parents=True, exist_ok=True)   

logging.info(f"loading model {model_name}...")
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
logging.info("loaded model successfully!")

config = model.config.to_dict()
with open(output_dir / "config.json", "w") as f:
    json.dump(config, f)
logging.info(f"config saved successfully! {config}")

state_dict = model.state_dict()
weights = {k: v.cpu().numpy() for k, v in state_dict.items()}

np.savez(output_dir / "weights.npz", weights)
torch.save(state_dict, output_dir / "weights.pt")
logging.info("weights saved successfully!")

dummy = torch.tensor([[464, 3290, 318, 257, 922]])  # "The dog is a good"
with torch.no_grad():
    out = model(dummy)
    logits = out.logits[0, -1] # [vocab_size], last position
 
torch.save(dummy, output_dir / "input_ids.pt")
torch.save(logits, output_dir / "logits.pt")
logging.info(f"input_ids shape: {dummy.shape}") 
logging.info(f"logits shape: {logits.shape}")

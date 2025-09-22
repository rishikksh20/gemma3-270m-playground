import torch
import os
from llm import Gemma3Model
from pathlib import Path
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from modules.mapping import load_weights_into_gemma
from modules.sampling import advance_decoding
from modules.tokenizer import GemmaTokenizer



def apply_chat_template(user_text):
    return f"<start_of_turn>user\n{user_text}<end_of_turn>\n<start_of_turn>model\n"


def test_gemma3_270M(prompt, config):

    model = Gemma3Model(dim=config["emb_dim"], depth=config["n_layers"], n_heads=config["n_heads"],
                       num_groups=config["n_kv_groups"], head_dim=config["head_dim"],
                       mlp_dim=config["hidden_dim"],
                       vocab_size=config["vocab_size"], context_length=config["context_length"],
                       window_size=config["sliding_window"], layer_types=config["layer_types"], dtype=config["dtype"])
    device = torch.device("cpu")



    repo_id = f"google/gemma-3-270m-it"

    local_dir = Path(repo_id).parts[-1]
    print("Download the model ...")
    weights_file = hf_hub_download(
            repo_id=repo_id,
            filename="model.safetensors",
            local_dir=local_dir,
        )
    weights_dict = load_file(weights_file)


    load_weights_into_gemma(model, config, weights_dict)
    model.to(device)
    del weights_dict

    hf_hub_download(
        repo_id=repo_id,
        filename="tokenizer.json",
        local_dir=local_dir,
    )
    tokenizer_file_path = os.path.join(local_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_file_path):
        try:
            tokenizer_file_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json", local_dir=local_dir)
        except Exception as e:
            print(f"Warning: failed to download tokenizer.json: {e}")
            tokenizer_file_path = "tokenizer.json"

    tokenizer = GemmaTokenizer(tokenizer_file_path=tokenizer_file_path)


    print(f"Prompt : {prompt}")
    prompt = "Give me a short introduction to large language models."
    prompt = apply_chat_template("Give me a short introduction to large language models.")
    input_token_ids = tokenizer.encode(prompt)
    text = tokenizer.decode(input_token_ids)
    print(f"Decoded Text: {text}")

    input_token_ids_tensor = torch.tensor(input_token_ids, device=device).unsqueeze(0)

    for token in advance_decoding(
            model=model,
            token_ids=input_token_ids_tensor,
            max_new_tokens=8192,
            eos_token_id=tokenizer.encode("<end_of_turn>")[-1],
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.2,
            window_size=50
    ):
        token_id = token.squeeze(0).tolist()
        print(
            tokenizer.decode(token_id),
            end="",
            flush=True
        )

if __name__ == "__main__":
    prompt = "Please explain the climate change and how it impacts our future."

    GEMMA270M_CONFIG = {
        "vocab_size": 262_144,  # Vocabulary size
        "context_length": 32_768,  # Context length that was used to train the model
        "emb_dim": 640,  # Embedding dimension
        "n_heads": 4,  # Number of attention heads
        "n_layers": 18,  # Number of layers
        "hidden_dim": 2048,  # Size of the intermediate dimension in FeedForward
        "head_dim": 256,  # Size of the heads in GQA
        "qk_norm": True,  # Whether to normalize queries and keys in GQA
        "n_kv_groups": 1,  # Key-Value groups for grouped-query attention
        "rope_base": 1_000_000.0,  # The base in RoPE's "theta"
        "sliding_window": 512,  # Sliding window attention size
        "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention"
    ],
        "dtype": torch.bfloat16
    }
    test_gemma3_270M(prompt, GEMMA270M_CONFIG)
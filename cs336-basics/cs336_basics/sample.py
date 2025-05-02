import torch
import argparse
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM, softmax
from cs336_basics.utils import load_checkpoint, DEVICE


def sample(model: TransformerLM, tokenizer: Tokenizer, text: str, max_tokens: int, temperature: float = None, top_p: float = None, eos_token: str = "<|endoftext|>") -> str:
    inputs = tokenizer.encode(text)
    inputs = torch.tensor(inputs).long()
    assert len(inputs.shape) == 1, "inputs must be a 1D tensor"
    inputs = inputs.unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(inputs)
            logits = logits[0, -1, :]
            if temperature is not None:
                logits /= temperature
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(softmax(sorted_logits, -1), dim=-1)
                cutoff_index = None
                for i, prob in enumerate(cumulative_probs):
                    if prob > top_p:
                        cutoff_index = i + 1
                        break
                assert cutoff_index is not None, "top_p is too high"
                sorted_indices = sorted_indices[:cutoff_index]
                sorted_logits = sorted_logits[:cutoff_index]
                probs = softmax(sorted_logits, -1)
                sorted_idx = torch.multinomial(probs, num_samples=1)
                next_token = sorted_indices[sorted_idx]
            else:
                probs = softmax(logits, -1)
                next_token = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat([inputs, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer._inv_vocab[eos_token.encode("utf-8")]:
                break
    return tokenizer.decode(inputs[0].tolist())


def main():
    parser = argparse.ArgumentParser(description="Sample from a TransformerLM")
    parser.add_argument(
        "input", type=str, help="Prompt to start the sampling from"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--tokenizer_path", type=str, required=True, help="Path to the tokenizer"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=100, help="Maximum number of tokens to sample"
    )
    parser.add_argument(
        "--temperature", type=float, default=None, help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", type=float, default=None, help="Top-p sampling cutoff"
    )
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the transformer model")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Dimension of the feedforward network")
    parser.add_argument("--attn_pdrop", type=float, default=0.05, help="Attention dropout rate")
    parser.add_argument("--residual_pdrop", type=float, default=0.1, help="Residual dropout rate")
    parser.add_argument("--context_length", type=int, default=256, help="Context length")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(f"{args.tokenizer_path}/vocab.pkl", f"{args.tokenizer_path}/merges.pkl", ["<|endoftext|>"])

    model = TransformerLM(
        len(tokenizer._vocab),
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
    ).to(DEVICE)

    load_checkpoint(args.model_path, model)

    generated_text = sample(
        model, tokenizer, args.input, args.max_tokens, args.temperature, args.top_p
    )
    print(generated_text)


if __name__ == "__main__":
    main()
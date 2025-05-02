import torch
import wandb
import logging
import argparse
import numpy as np
from tqdm import tqdm

from cs336_basics.loss import cross_entropy
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.utils import get_batch, save_checkpoint, load_checkpoint, DEVICE
from cs336_basics.optimizer import AdamW, learning_rate_schedule, gradient_clipping

# setup logging
logging.basicConfig(format="%(asctime)s (%(levelname)s): %(message)s")
logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        run_name: str,
        tokenizer: Tokenizer,
        model: TransformerLM,
        train_dataset: np.ndarray,
        val_dataset: np.ndarray,
        learning_rate: float,
        use_lr_schedule: bool,
        num_iters: int,
        val_every: int,
        checkpoint_every: int,
        warmup_iters: int,
        betas: tuple[float, float],
        epsilon: float,
        weight_decay: float,
        cosine_cycle_iters: int,
        min_learning_rate: float,
        max_grad_norm: float,
        batch_size: int,
        context_length: int,
        val_iters: int
    ):
        self.run_name = run_name
        self.tokenizer = tokenizer
        self.model = model
        self.optimizer = AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=epsilon
        )
        self.lr = learning_rate
        self.min_learning_rate = min_learning_rate
        self.use_lr_schedule = use_lr_schedule
        self.warmup_iters = warmup_iters
        self.cosine_cycle_iters = cosine_cycle_iters
        self.max_grad_norm = max_grad_norm
        self.num_iters = num_iters
        self.val_every = val_every
        self.checkpoint_every = checkpoint_every
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.context_length = context_length
        self.val_iters = val_iters
        self.betas = betas
        self.epsilon = epsilon
        self.weight_decay = weight_decay

    def train(self, iteration: int = 0):
        wandb.init(
            project="cs336",
            name=self.run_name,
            config={
                "learning_rate": self.lr,
                "use_lr_schedule": self.use_lr_schedule,
                "num_iters": self.num_iters,
                "val_every": self.val_every,
                "checkpoint_every": self.checkpoint_every,
                "warmup_iters": self.warmup_iters,
                "cosine_cycle_iters": self.cosine_cycle_iters,
                "max_grad_norm": self.max_grad_norm,
                "min_learning_rate": self.min_learning_rate,
                "batch_size": self.batch_size,
                "context_length": self.context_length,
                "val_iters": self.val_iters,
                "betas": self.betas,
                "epsilon": self.epsilon,
                "weight_decay": self.weight_decay
            }
        )
        wandb.watch(self.model, log="all")

        for iteration in tqdm(range(iteration, self.num_iters)):
            if self.use_lr_schedule:
                lr = learning_rate_schedule(
                    iteration,
                    max_learning_rate=self.lr,
                    min_learning_rate=self.min_learning_rate,
                    warmup_iters=self.warmup_iters,
                    cosine_cycle_iters=self.cosine_cycle_iters,
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            self.model.train()
            values, targets = get_batch(
                self.train_dataset, batch_size=self.batch_size, context_length=self.context_length, device=DEVICE
            )
            if len(values.shape) == 1:
                values = values.unsqueeze(0)
                targets = targets.unsqueeze(0)
            self.optimizer.zero_grad()
            logits = self.model(values)
            loss = cross_entropy(logits, targets)
            loss.backward()
            gradient_clipping(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            wandb.log({"train_loss": loss.item(), "learning_rate": self.optimizer.param_groups[0]["lr"]})
            logger.debug(f"Iteration {iteration}: train_loss={loss.item()}")

            if (iteration + 1) % self.val_every == 0:
                self.model.eval()
                val_loss = 0
                perplexity = 0
                with torch.no_grad():
                    for _ in range(self.val_iters):
                        val_values, val_targets = get_batch(
                            self.val_dataset, batch_size=self.batch_size, context_length=self.context_length, device=DEVICE
                        )
                        if len(val_values.shape) == 1:
                            val_values = val_values.unsqueeze(0)
                            val_targets = val_targets.unsqueeze(0)
                        val_logits = self.model(val_values)
                        loss = cross_entropy(val_logits, val_targets)
                        val_loss += loss.item()
                        perplexity += torch.exp(loss).item()
                val_loss /= self.val_iters
                perplexity /= self.val_iters
                wandb.log({"val_loss": val_loss, "perplexity": perplexity})
                logger.debug(f"Iteration {iteration}: val_loss={val_loss}")

            if (iteration + 1) % self.checkpoint_every == 0:
                save_checkpoint(
                    self.model, self.optimizer, iteration, f"checkpoints/{self.run_name}/{iteration}"
                )
                logger.debug(f"Saved checkpoint at iteration {iteration}")

        wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train a TransformerLM model")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run")
    parser.add_argument("--train_path", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("--val_path", type=str, required=True, help="Path to the validation dataset")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the vocabulary")
    parser.add_argument("--cosine_cycle_iters", type=int, required=True, help="Number of iterations in the cosine cycle")
    parser.add_argument("--min_learning_rate", type=float, required=True, help="Minimum learning rate")
    parser.add_argument("--num_iters", type=int, required=True, help="Number of iterations")
    parser.add_argument("--val_every", type=int, required=True, help="Validate every N iterations")
    parser.add_argument("--checkpoint_every", type=int, required=True, help="Checkpoint every N iterations")
    parser.add_argument("--warmup_iters", type=int, required=True, help="Number of warmup iterations")
    parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of layers in the transformer model")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model")
    parser.add_argument("--num_heads", type=int, default=16, help="Number of attention heads")
    parser.add_argument("--d_ff", type=int, default=2048, help="Dimension of the feedforward network")
    parser.add_argument("--attn_pdrop", type=float, default=0.05, help="Attention dropout rate")
    parser.add_argument("--residual_pdrop", type=float, default=0.1, help="Residual dropout rate")
    parser.add_argument("--model-checkpoint", default=None, type=str, help="Path to a model checkpoint")
    parser.add_argument("--use_lr_schedule", action="store_true", default=True, help="Use a learning rate schedule")
    parser.add_argument("--epsilon", type=float, default=1e-8, help="AdamW epsilon")
    parser.add_argument("--beta1", type=float, default=0.9, help="AdamW beta1")
    parser.add_argument("--beta2", type=float, default=0.999, help="AdamW beta2")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--context_length", type=int, default=256, help="Context length")
    parser.add_argument("--val_iters", type=int, default=100, help="Number of validation iterations")
    parser.add_argument("--log_level", type=str, default="debug", help="Logging level")
    parser.add_argument("--is_parallel", action="store_true", default=False, help="Use parallel transformer blocks")
    parser.add_argument("--norm_type", type=str, default="pre", help="Type of normalization to use (pre, post, none)")

    args = parser.parse_args()

    logger.setLevel(args.log_level.upper())

    tokenizer = Tokenizer.from_files(f"{args.tokenizer_path}/vocab.pkl", f"{args.tokenizer_path}/merges.pkl")
    logger.debug(f"Loaded tokenizer with {tokenizer.vocab_size} tokens")
    train_dataset = np.memmap(args.train_path, mode="r", dtype=np.uint16)
    val_dataset = np.memmap(args.val_path, mode="r", dtype=np.uint16)
    logger.debug("Loaded datasets")

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
        is_parallel=args.is_parallel,
        norm_type=args.norm_type
    ).to(DEVICE)
    logger.debug("Initialized model")

    trainer = Trainer(
        run_name=args.run_name,
        tokenizer=tokenizer,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=args.learning_rate,
        use_lr_schedule=args.use_lr_schedule,
        num_iters=args.num_iters,
        val_every=args.val_every,
        checkpoint_every=args.checkpoint_every,
        warmup_iters=args.warmup_iters,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        cosine_cycle_iters=args.cosine_cycle_iters,
        min_learning_rate=args.min_learning_rate,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        context_length=args.context_length,
        val_iters=args.val_iters,
        epsilon=args.epsilon
    )

    if args.model_checkpoint:
        iteration = load_checkpoint(args.model_checkpoint, model, trainer.optimizer)
    else:
        iteration = 0

    logger.info(f"Starting training from iteration {iteration}")
    trainer.train(iteration)
    logger.info("Training complete")
    
    save_checkpoint(
        model, trainer.optimizer, iteration, f"checkpoints/{args.run_name}/final"
    )
    logger.info("Saved final checkpoint")

if __name__ == "__main__":
    main()
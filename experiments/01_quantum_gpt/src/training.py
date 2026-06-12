import os
import json
import time
import torch
from datetime import datetime
from tqdm import tqdm
from dataclasses import asdict
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchviz import make_dot
from src.dataset import InputDataset
from src.model import QuantumGPT


class Trainer:
    def __init__(self, config, model_name, dataset_name):
        self.config = config
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.run_dir = self._setup_run_dir()
        self.writer = SummaryWriter(log_dir=self.run_dir)
        self.dataset = InputDataset(config, dataset_name)
        self.model = QuantumGPT(config, self.dataset.tokenizer.vocab_size).to(config.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.learning_rate
        )

    def _setup_run_dir(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("experiments", f"{self.model_name}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def _log_architecture(self):
        dummy_input = torch.zeros((1, self.config.block_size), dtype=torch.long).to(
            self.config.device
        )
        try:
            self.writer.add_graph(self.model, dummy_input)
        except Exception as e:
            print(f"Could not log model graph to TensorBoard: {e}")

        try:
            logits, _ = self.model(dummy_input)
            dot = make_dot(logits, params=dict(self.model.named_parameters()))
            dot.format = "png"
            dot.render(os.path.join(self.run_dir, "model_architecture_graph"))
        except Exception as e:
            print(f"Could not generate torchviz graph: {e}")

        with open(os.path.join(self.run_dir, "model_summary_table.txt"), "w") as f:
            f.write(str(summary(self.model, input_data=dummy_input, verbose=0)))

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ["train", "val"]:
            losses = torch.zeros(self.config.eval_iters)
            for k in range(self.config.eval_iters):
                X, Y = self.dataset.get_batch(split, self.config.batch_size)
                _, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        self.model.train()
        return out

    def train(self):
        print(f"\n--- Starting Training | Run: {self.run_dir} ---")
        print(f"Dataset: {self.dataset_name} | Tokenizer: {self.config.tokenizer_class}")

        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=4)
        with open(os.path.join(self.run_dir, "dictionary.json"), "w") as f:
            json.dump(self.dataset.tokenizer.to_dict(), f, indent=4)

        self._log_architecture()

        best_val_loss = float("inf")
        history = []
        start_time = time.time()
        pbar = tqdm(range(self.config.max_iters), desc="Training Progress")

        for iter in pbar:
            if iter % self.config.eval_interval == 0 or iter == self.config.max_iters - 1:
                losses = self.estimate_loss()
                self.writer.add_scalar("Loss/train", losses["train"], iter)
                self.writer.add_scalar("Loss/val", losses["val"], iter)
                history.append({
                    "step": iter,
                    "train_loss": round(losses["train"], 4),
                    "val_loss": round(losses["val"], 4),
                })
                pbar.set_description(f"Step {iter} | Val Loss: {losses['val']:.4f}")

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    torch.save(
                        self.model.state_dict(),  # nosec B614
                        os.path.join(self.run_dir, "best_model.pth"),
                    )

            xb, yb = self.dataset.get_batch("train", self.config.batch_size)
            _, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if iter % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        total_duration = time.time() - start_time

        self.writer.add_hparams(
            asdict(self.config),
            {"hparam/best_val_loss": best_val_loss, "hparam/train_time": total_duration},
        )

        metrics = {
            "model_name": self.model_name,
            "dataset": self.dataset_name,
            "tokenizer": self.config.tokenizer_class,
            "total_training_time_seconds": round(total_duration, 2),
            "best_val_loss": round(best_val_loss, 4),
            "history": history,
        }
        with open(os.path.join(self.run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        torch.save(
            self.model.state_dict(),
            os.path.join(self.run_dir, "final_model.pth"),  # nosec B614
        )
        self.writer.close()

        print(f"\nTraining completed in {total_duration:.2f}s.")
        return self.run_dir

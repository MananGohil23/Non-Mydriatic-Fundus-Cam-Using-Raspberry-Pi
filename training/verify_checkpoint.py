import argparse

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    print("top-level keys:", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__)
    if not isinstance(ckpt, dict):
        return

    if "args" in ckpt:
        pre = ckpt["args"]
        print("pretrain input_size:", getattr(pre, "input_size", None))
        print("pretrain model:", getattr(pre, "model", None))
        print("pretrain epochs:", getattr(pre, "epochs", None))

    state = ckpt.get("model", {})
    print("model state entries:", len(state))
    print("has head:", any(k.startswith("head") for k in state))
    print("has fc_norm:", any(k.startswith("fc_norm") for k in state))
    print("has final norm:", any(k == "norm.weight" for k in state))
    print("encoder blocks:", sum(1 for k in state if k.startswith("blocks.")))
    print("decoder entries:", sum(1 for k in state if k.startswith("decoder")))
    print("pos_embed shape:", tuple(state["pos_embed"].shape) if "pos_embed" in state else None)
    print("epoch:", ckpt.get("epoch"))


if __name__ == "__main__":
    main()

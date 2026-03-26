import pickle
import sys
import numpy as np
import pdb


def print_struct(obj, indent=0, max_items=5):
    prefix = "  " * indent
    if isinstance(obj, dict):
        print(f"{prefix}dict ({len(obj)} keys):")
        for k, v in obj.items():
            print(f"{prefix}  [{repr(k)}]:")
            print_struct(v, indent + 2)
    elif isinstance(obj, (list, tuple)):
        tname = type(obj).__name__
        print(f"{prefix}{tname} ({len(obj)} items):")
        for i, v in enumerate(obj[:max_items]):
            print(f"{prefix}  [{i}]:")
            print_struct(v, indent + 2)
        if len(obj) > max_items:
            print(f"{prefix}  ... ({len(obj) - max_items} more)")
    elif isinstance(obj, np.ndarray):
        print(f"{prefix}ndarray shape={obj.shape} dtype={obj.dtype}")
    else:
        try:
            import torch
            if isinstance(obj, torch.Tensor):
                print(f"{prefix}Tensor shape={tuple(obj.shape)} dtype={obj.dtype} device={obj.device}")
                return
        except ImportError:
            pass
        if hasattr(obj, "__dict__"):
            tname = type(obj).__name__
            print(f"{prefix}{tname}:")
            for k, v in vars(obj).items():
                print(f"{prefix}  .{k}:")
                print_struct(v, indent + 2)
        else:
            print(f"{prefix}{type(obj).__name__}: {repr(obj)[:120]}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python read_pkl.py <file.pkl>")
        sys.exit(1)

    path = sys.argv[1]
    with open(path, "rb") as f:
        data = pickle.load(f)

    pdb.set_trace()  # Set a breakpoint to inspect the data structure
    print(f"=== {path} ===")
    print_struct(data)


if __name__ == "__main__":
    main()

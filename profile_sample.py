import sys
import os
import torch
import cProfile
import pstats
from io import StringIO
from run.TrainingModule import TrainingModule
from model.model import TreeDenoiserTokenGT
from utils.random_tree import Tree


# Create a dummy dataset class just in case, though we pass None
class DummyDataset:
    pass


def profile_sample():
    print("Setting up profiling...")

    # 1. Initialize Model
    # Using small dimensions for speed, just profiling the logic overhead mostly
    model = TreeDenoiserTokenGT(
        num_node_types=100,  # arbitrary
        num_edge_types=100,  # arbitrary
        embed_dim=64,
        phyla_dim=64,
        n_layers=2,
        n_heads=2,
        output_dim=1,
    )

    # 2. Initialize TrainingModule
    tm = TrainingModule(model=model, dataset=None, verbose=False)  # explicit None
    tm.eval()
    tm.to("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Prepare Input
    # Generate a random tree
    num_leaves = 10
    rt = Tree(num_leaves=num_leaves, random=True)
    newick_str = str(rt)

    # Dummy embeddings: (B, N, D) -> (1, num_leaves, embed_dim)
    # The dimensions usually match the model's embed_dim if used directly,
    # but let's check how they are used.
    # In TrainingModule.forward (which calls model), it seems to expect some embeddings.
    # Looking at TrainingModule code:
    # v_pred, edge_split_masks, edge_mask = self.forward(tokenized, time, embeddings)
    # The embeddings are likely phyla_embeddings.
    # Let's assume size (1, num_leaves, embed_dim)

    phyla_embeddings = torch.randn(1, num_leaves, 64).to(tm.device)

    print("Starting profiling...")
    pr = cProfile.Profile()
    pr.enable()

    # 4. Run sample
    # T=0.1, dt_base=0.02 -> approx 5 steps
    try:
        with torch.no_grad():
            tm.sample(
                newick_starting_trees=[newick_str],
                phyla_embeddings=phyla_embeddings,
                T=0.1,
                dt_base=0.01,
                max_steps=50,  # Cap steps to ensure it finishes quickly
            )
    except Exception as e:
        print(f"Error during sampling: {e}")
        # print stack trace
        import traceback

        traceback.print_exc()

    pr.disable()
    print("Profiling finished.")

    # 5. Print results
    s = StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(30)  # Print top 30
    print(s.getvalue())


if __name__ == "__main__":
    profile_sample()

import os

# Set non-interactive backend before any matplotlib imports in CI
if os.environ.get("CI"):
    import matplotlib

    matplotlib.use("Agg")

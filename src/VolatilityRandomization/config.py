import os

# Default output root — relative to current working directory
_output_root = os.environ.get("VR_OUTPUT_ROOT", os.path.join(os.getcwd(), "outputs"))

def get_output_dir(kind: str) -> str:
    """
    Returns the output directory path for a given kind ("logs", "calibrations", etc.).
    Ensures the folder exists.
    """
    path = os.path.join(_output_root, kind)
    os.makedirs(path, exist_ok=True)
    return path
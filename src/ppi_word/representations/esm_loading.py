"""Load the official ESM-2 checkpoint for residue representations."""
import warnings

import torch


def load_esm2_representations(checkpoint):
    """Return the 33-layer ESM-2 model and alphabet from a local checkpoint.

    Input is the trusted official esm2_t33_650M_UR50D.pt file. The fair-esm
    core loader verifies keys/shapes while allowing only absent contact-head
    regression weights. No companion file or network download is used.
    Call with return_contacts=False: contact prediction is outside this loader's
    scope. The returned model is on CPU; the caller sets evaluation mode/dtype.
    """
    from esm.pretrained import load_model_and_alphabet_core

    # Official weights include a pickled configuration; mmap avoids an extra
    # resident copy of the large weight tensors while constructing the model.
    data = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
    config = data["cfg"]["model"]
    if config.encoder_layers != 33 or config.encoder_embed_dim != 1280:
        raise ValueError("esm2_t33_650M_UR50D weights are required")
    with warnings.catch_warnings():
        # This warning concerns the unused head, not residue representations.
        warnings.filterwarnings(
            "ignore", category=UserWarning, module=r"esm\.pretrained",
            message=r"Regression weights not found, predicting contacts will not produce correct results\.",
        )
        return load_model_and_alphabet_core("esm2_t33_650M_UR50D", data, regression_data=None)

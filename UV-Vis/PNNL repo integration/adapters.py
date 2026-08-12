from dtnn_adapter import predict_dtnn
from mpnn_adapter import predict_mpnn
from schnet_adapter import predict_schnet
from transformer_adapter import predict_transformer


ADAPTERS = {
    "DTNN": predict_dtnn,
    "MPNN": predict_mpnn,
    "SchNet": predict_schnet,
    "Transformer": predict_transformer
}


def run_pnnl_model(
        model_name,
        smiles
):
    if model_name not in ADAPTERS:
        raise ValueError(
            f"Unknown PNNL model: {model_name}"
        )

    return ADAPTERS[
        model_name
    ](
        smiles
    )


def run_pnnl_models(
        model_names,
        smiles
):
    return [
        run_pnnl_model(
            model_name=model_name,
            smiles=smiles
        )
        for model_name in model_names
    ]
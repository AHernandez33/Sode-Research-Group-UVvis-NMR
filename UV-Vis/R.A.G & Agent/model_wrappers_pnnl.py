from adapters import ADAPTERS


class PNNLWorkflowWrapper:
    def __init__(
        self,
        model_name
    ):
        if model_name not in ADAPTERS:
            raise ValueError(
                f"No PNNL adapter exists for {model_name}."
            )

        self.model_name = model_name

        self.adapter = ADAPTERS[
            model_name
        ]

    def predict(
        self,
        smiles,
        model_features=None
    ):
        return self.adapter(
            smiles
        )


def build_pnnl_wrappers():
    wrappers = {}

    for model_name in ADAPTERS:
        wrappers[
            model_name
        ] = PNNLWorkflowWrapper(
            model_name
        )

    return wrappers
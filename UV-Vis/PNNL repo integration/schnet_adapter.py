from pathlib import Path

from original_config import schnet_folder

from common import build_failed_result


def predict_schnet(
        smiles,
        normalize=True
):
    try:
        schnet_path = Path(
            schnet_folder
        )

        if not schnet_path.exists():
            raise FileNotFoundError(
                f"PNNL SchNet folder was not found: {schnet_path}"
            )

        raise NotImplementedError(
            "The SchNet adapter is registered correctly, but the PNNL "
            "SchNet inference workflow has not been connected yet. "
            "Create a predict_agent.py script inside UVvis-SchNet, then "
            "update predict_schnet() to call it."
        )

    except Exception as error:
        return build_failed_result(
            model_name="schnet",
            error=error
        )


def main():
    smiles = input(
        "Enter a SMILES string: "
    ).strip()

    prediction = predict_schnet(
        smiles
    )

    print(
        prediction
    )


if __name__ == "__main__":
    main()
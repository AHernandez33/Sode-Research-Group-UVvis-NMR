import json

from model_wrappers_pnnl import build_pnnl_wrappers


def main():
    smiles = input(
        "Enter a SMILES string: "
    ).strip()

    wrappers = build_pnnl_wrappers()

    for model_name, wrapper in wrappers.items():
        print()
        print(
            f"Testing {model_name}"
        )

        prediction = wrapper.predict(
            smiles=smiles
        )

        print(
            json.dumps(
                prediction,
                indent=4
            )
        )


if __name__ == "__main__":
    main()

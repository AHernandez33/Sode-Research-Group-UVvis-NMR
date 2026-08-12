from pathlib import Path
import sys


integration_folder = Path(
    __file__
).resolve().parent

uv_vis_folder = integration_folder.parent


wavelength_min = 220
wavelength_max = 400

wavelengths = list(
    range(
        wavelength_min,
        wavelength_max + 1
    )
)


python_executable = str(
    Path(
        sys.executable
    ).resolve()
)

command_timeout_seconds = 2400


dtnn_folder = (
    uv_vis_folder
    / "dtnn"
)

dtnn_predict_file = (
    dtnn_folder
    / "predict.py"
)

dtnn_predict_xyz_file = (
    dtnn_folder
    / "predict_xyz.py"
)


mpnn_root_folder = (
    uv_vis_folder
    / "mpnn"
)

mpnn_folder = (
    mpnn_root_folder
    / "3D_only"
)

mpnn_predict_file = (
    mpnn_folder
    / "predict.py"
)

mpnn_checkpoint_folder = (
    mpnn_folder
    / "models_3D_only"
)


def find_checkpoint_folder(
        root_folder
):
    root_folder = Path(
        root_folder
    ).resolve()

    checkpoint_candidates = [
        root_folder
        / "models_3D_only",

        root_folder
        / "model_checkpoints",

        root_folder
        / "checkpoints",

        root_folder
        / "models",

        root_folder
        / "trained_models",

        root_folder
        / "weights",

        root_folder
        / "model_weights"
    ]

    for checkpoint_candidate in checkpoint_candidates:
        if checkpoint_candidate.exists():
            checkpoint_files = list(
                checkpoint_candidate.rglob(
                    "*.pt"
                )
            )

            checkpoint_files.extend(
                checkpoint_candidate.rglob(
                    "*.pth"
                )
            )

            if checkpoint_files:
                return checkpoint_candidate.resolve()

    checkpoint_files = list(
        root_folder.rglob(
            "*.pt"
        )
    )

    checkpoint_files.extend(
        root_folder.rglob(
            "*.pth"
        )
    )

    if checkpoint_files:
        return checkpoint_files[
            0
        ].parent.resolve()

    return (
        root_folder
        / "model_checkpoints"
    ).resolve()


mpnn_checkpoint_folder = find_checkpoint_folder(
    mpnn_folder
)


schnet_folder = (
    uv_vis_folder
    / "schnet"
)

schnet_predict_file = (
    schnet_folder
    / "predict_agent.py"
)


transformer_folder = (
    uv_vis_folder
    / "transformer"
)

transformer_predict_file = (
    transformer_folder
    / "predict.py"
)


def find_transformer_model(
        root_folder
):
    root_folder = Path(
        root_folder
    ).resolve()

    if not root_folder.exists():
        return None

    model_files = list(
        root_folder.rglob(
            "*.model"
        )
    )

    if model_files:
        trained_model_files = [
            model_file
            for model_file in model_files
            if "trained" in model_file.name.lower()
        ]

        if trained_model_files:
            trained_model_files.sort(
                key=lambda model_file: model_file.stat().st_mtime,
                reverse=True
            )

            return trained_model_files[
                0
            ].resolve()

        model_files.sort(
            key=lambda model_file: model_file.stat().st_mtime,
            reverse=True
        )

        return model_files[
            0
        ].resolve()

    checkpoint_files = list(
        root_folder.rglob(
            "*.pt"
        )
    )

    checkpoint_files.extend(
        root_folder.rglob(
            "*.pth"
        )
    )

    if checkpoint_files:
        checkpoint_files.sort(
            key=lambda checkpoint_file: checkpoint_file.stat().st_mtime,
            reverse=True
        )

        return checkpoint_files[
            0
        ].resolve()

    return None


transformer_model_file = find_transformer_model(
    transformer_folder
)


def find_model_files(
        root_folder
):
    root_folder = Path(
        root_folder
    ).resolve()

    if not root_folder.exists():
        return []

    model_files = []

    for extension in [
        "*.pt",
        "*.pth",
        "*.model"
    ]:
        model_files.extend(
            root_folder.rglob(
                extension
            )
        )

    return [
        model_file.resolve()
        for model_file in model_files
    ]


def print_configuration():
    print()
    print(
        "PNNL configuration"
    )
    print(
        "------------------"
    )

    print(
        "Integration folder:",
        integration_folder
    )

    print(
        "UV-Vis folder:",
        uv_vis_folder
    )

    print(
        "Python executable:",
        python_executable
    )

    print(
        "Python executable exists:",
        Path(
            python_executable
        ).exists()
    )

    print()
    print(
        "DTNN folder:",
        dtnn_folder
    )

    print(
        "DTNN folder exists:",
        dtnn_folder.exists()
    )

    print(
        "DTNN predict.py:",
        dtnn_predict_file
    )

    print(
        "DTNN predict.py exists:",
        dtnn_predict_file.exists()
    )

    print(
        "DTNN predict_xyz.py:",
        dtnn_predict_xyz_file
    )

    print(
        "DTNN predict_xyz.py exists:",
        dtnn_predict_xyz_file.exists()
    )

    dtnn_model_files = find_model_files(
        dtnn_folder
    )

    print(
        "DTNN model files found:",
        len(
            dtnn_model_files
        )
    )

    for model_file in dtnn_model_files[
        :10
    ]:
        print(
            " -",
            model_file
        )

    print()
    print(
        "MPNN root folder:",
        mpnn_root_folder
    )

    print(
        "MPNN folder:",
        mpnn_folder
    )

    print(
        "MPNN folder exists:",
        mpnn_folder.exists()
    )

    print(
        "MPNN predict.py:",
        mpnn_predict_file
    )

    print(
        "MPNN predict.py exists:",
        mpnn_predict_file.exists()
    )

    print(
        "MPNN checkpoint folder:",
        mpnn_checkpoint_folder
    )

    print(
        "MPNN checkpoint folder exists:",
        mpnn_checkpoint_folder.exists()
    )

    mpnn_model_files = find_model_files(
        mpnn_folder
    )

    print(
        "MPNN model files found:",
        len(
            mpnn_model_files
        )
    )

    for model_file in mpnn_model_files[
        :20
    ]:
        print(
            " -",
            model_file
        )

    print()
    print(
        "SchNet folder:",
        schnet_folder
    )

    print(
        "SchNet folder exists:",
        schnet_folder.exists()
    )

    print(
        "SchNet predict_agent.py:",
        schnet_predict_file
    )

    print(
        "SchNet predict_agent.py exists:",
        schnet_predict_file.exists()
    )

    schnet_model_files = find_model_files(
        schnet_folder
    )

    print(
        "SchNet model files found:",
        len(
            schnet_model_files
        )
    )

    for model_file in schnet_model_files[
        :10
    ]:
        print(
            " -",
            model_file
        )

    print()
    print(
        "Transformer folder:",
        transformer_folder
    )

    print(
        "Transformer folder exists:",
        transformer_folder.exists()
    )

    print(
        "Transformer predict.py:",
        transformer_predict_file
    )

    print(
        "Transformer predict.py exists:",
        transformer_predict_file.exists()
    )

    print(
        "Transformer model file:",
        transformer_model_file
    )

    print(
        "Transformer model file exists:",
        (
            transformer_model_file is not None
            and transformer_model_file.exists()
        )
    )

    transformer_model_files = find_model_files(
        transformer_folder
    )

    print(
        "Transformer model files found:",
        len(
            transformer_model_files
        )
    )

    for model_file in transformer_model_files[
        :10
    ]:
        print(
            " -",
            model_file
        )

    print()
    print(
        "Wavelength count:",
        len(
            wavelengths
        )
    )

    print(
        "Wavelength range:",
        wavelengths[
            0
        ],
        "to",
        wavelengths[
            -1
        ]
    )


if __name__ == "__main__":
    print_configuration()

# from pathlib import Path
# import sys


# integration_folder = Path(
#     __file__
# ).resolve().parent

# uv_vis_folder = integration_folder.parent


# wavelength_min = 220
# wavelength_max = 400

# wavelengths = list(
#     range(
#         wavelength_min,
#         wavelength_max + 1
#     )
# )


# python_executable = str(
#     Path(
#         sys.executable
#     ).resolve()
# )

# command_timeout_seconds = 2400


# dtnn_folder = (
#     uv_vis_folder
#     / "dtnn"
# )

# dtnn_predict_file = (
#     dtnn_folder
#     / "predict.py"
# )

# dtnn_predict_xyz_file = (
#     dtnn_folder
#     / "predict_xyz.py"
# )


# mpnn_root_folder = (
#     uv_vis_folder
#     / "mpnn"
# )

# mpnn_folder = (
#     mpnn_root_folder
#     / "3D_only"
# )

# mpnn_predict_file = (
#     mpnn_folder
#     / "predict.py"
# )

# mpnn_checkpoint_folder = (
#     mpnn_folder
#     / "models_3D_only"
# )

# def find_checkpoint_folder(
#         root_folder
# ):
#     root_folder = Path(
#         root_folder
#     )

#     checkpoint_candidates = [
#         root_folder
#         / "model_checkpoints",

#         root_folder
#         / "checkpoints",

#         root_folder
#         / "models",

#         root_folder
#         / "trained_models",

#         root_folder
#         / "weights",

#         root_folder
#         / "model_weights"
#     ]

#     for checkpoint_candidate in checkpoint_candidates:
#         if checkpoint_candidate.exists():
#             checkpoint_files = list(
#                 checkpoint_candidate.rglob(
#                     "*.pt"
#                 )
#             )

#             checkpoint_files.extend(
#                 checkpoint_candidate.rglob(
#                     "*.pth"
#                 )
#             )

#             if checkpoint_files:
#                 return checkpoint_candidate

#     checkpoint_files = list(
#         root_folder.rglob(
#             "*.pt"
#         )
#     )

#     checkpoint_files.extend(
#         root_folder.rglob(
#             "*.pth"
#         )
#     )

#     if checkpoint_files:
#         checkpoint_parent_folders = [
#             checkpoint_file.parent
#             for checkpoint_file in checkpoint_files
#         ]

#         common_parent = Path(
#             checkpoint_parent_folders[
#                 0
#             ]
#         )

#         while not all(
#             common_parent in folder.parents
#             or common_parent == folder
#             for folder in checkpoint_parent_folders
#         ):
#             if common_parent == root_folder:
#                 break

#             common_parent = common_parent.parent

#         return common_parent

#     return (
#         root_folder
#         / "model_checkpoints"
#     )


# mpnn_checkpoint_folder = find_checkpoint_folder(
#     mpnn_folder
# )


# schnet_folder = (
#     uv_vis_folder
#     / "schnet"
# )

# schnet_predict_file = (
#     schnet_folder
#     / "predict_agent.py"
# )


# transformer_folder = (
#     uv_vis_folder
#     / "transformer"
# )

# transformer_predict_file = (
#     transformer_folder
#     / "predict.py"
# )


# def find_transformer_model(
#         root_folder
# ):
#     root_folder = Path(
#         root_folder
#     )

#     model_files = list(
#         root_folder.rglob(
#             "*.model"
#         )
#     )

#     if model_files:
#         return model_files[
#             0
#         ]

#     checkpoint_files = list(
#         root_folder.rglob(
#             "*.pt"
#         )
#     )

#     checkpoint_files.extend(
#         root_folder.rglob(
#             "*.pth"
#         )
#     )

#     if checkpoint_files:
#         return checkpoint_files[
#             0
#         ]

#     return None


# transformer_model_file = find_transformer_model(
#     transformer_folder
# )


# def find_model_files(
#         root_folder
# ):
#     root_folder = Path(
#         root_folder
#     )

#     if not root_folder.exists():
#         return []

#     model_files = []

#     for extension in [
#         "*.pt",
#         "*.pth",
#         "*.model"
#     ]:
#         model_files.extend(
#             root_folder.rglob(
#                 extension
#             )
#         )

#     return model_files


# def print_configuration():
#     print()
#     print(
#         "PNNL configuration"
#     )
#     print(
#         "------------------"
#     )

#     print(
#         "Integration folder:",
#         integration_folder
#     )

#     print(
#         "UV-Vis folder:",
#         uv_vis_folder
#     )

#     print(
#         "Python executable:",
#         python_executable
#     )

#     print()
#     print(
#         "DTNN folder:",
#         dtnn_folder
#     )

#     print(
#         "DTNN folder exists:",
#         dtnn_folder.exists()
#     )

#     print(
#         "DTNN predict.py:",
#         dtnn_predict_file
#     )

#     print(
#         "DTNN predict.py exists:",
#         dtnn_predict_file.exists()
#     )

#     print(
#         "DTNN predict_xyz.py:",
#         dtnn_predict_xyz_file
#     )

#     print(
#         "DTNN predict_xyz.py exists:",
#         dtnn_predict_xyz_file.exists()
#     )

#     dtnn_model_files = find_model_files(
#         dtnn_folder
#     )

#     print(
#         "DTNN model files found:",
#         len(
#             dtnn_model_files
#         )
#     )

#     for model_file in dtnn_model_files[
#         :10
#     ]:
#         print(
#             " -",
#             model_file
#         )

#     print()
#     print(
#         "MPNN root folder:",
#         mpnn_root_folder
#     )

#     print(
#         "MPNN folder:",
#         mpnn_folder
#     )

#     print(
#         "MPNN folder exists:",
#         mpnn_folder.exists()
#     )

#     print(
#         "MPNN predict.py:",
#         mpnn_predict_file
#     )

#     print(
#         "MPNN predict.py exists:",
#         mpnn_predict_file.exists()
#     )

#     print(
#         "MPNN checkpoint folder:",
#         mpnn_checkpoint_folder
#     )

#     print(
#         "MPNN checkpoint folder exists:",
#         mpnn_checkpoint_folder.exists()
#     )

#     mpnn_model_files = find_model_files(
#         mpnn_folder
#     )

#     print(
#         "MPNN model files found:",
#         len(
#             mpnn_model_files
#         )
#     )

#     for model_file in mpnn_model_files[
#         :20
#     ]:
#         print(
#             " -",
#             model_file
#         )

#     print()
#     print(
#         "SchNet folder:",
#         schnet_folder
#     )

#     print(
#         "SchNet folder exists:",
#         schnet_folder.exists()
#     )

#     print(
#         "SchNet predict_agent.py:",
#         schnet_predict_file
#     )

#     print(
#         "SchNet predict_agent.py exists:",
#         schnet_predict_file.exists()
#     )

#     schnet_model_files = find_model_files(
#         schnet_folder
#     )

#     print(
#         "SchNet model files found:",
#         len(
#             schnet_model_files
#         )
#     )

#     for model_file in schnet_model_files[
#         :10
#     ]:
#         print(
#             " -",
#             model_file
#         )

#     print()
#     print(
#         "Transformer folder:",
#         transformer_folder
#     )

#     print(
#         "Transformer folder exists:",
#         transformer_folder.exists()
#     )

#     print(
#         "Transformer predict.py:",
#         transformer_predict_file
#     )

#     print(
#         "Transformer predict.py exists:",
#         transformer_predict_file.exists()
#     )

#     print(
#         "Transformer model file:",
#         transformer_model_file
#     )

#     print(
#         "Transformer model file exists:",
#         (
#             transformer_model_file is not None
#             and transformer_model_file.exists()
#         )
#     )

#     transformer_model_files = find_model_files(
#         transformer_folder
#     )

#     print(
#         "Transformer model files found:",
#         len(
#             transformer_model_files
#         )
#     )

#     for model_file in transformer_model_files[
#         :10
#     ]:
#         print(
#             " -",
#             model_file
#         )

#     print()
#     print(
#         "Wavelength count:",
#         len(
#             wavelengths
#         )
#     )

#     print(
#         "Wavelength range:",
#         wavelengths[
#             0
#         ],
#         "to",
#         wavelengths[
#             -1
#         ]
#     )


# if __name__ == "__main__":
#     print_configuration()
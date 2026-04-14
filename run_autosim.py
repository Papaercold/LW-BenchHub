"""Run an autosim pipeline registered in lw_benchhub."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run an lw_benchhub autosim pipeline.")
parser.add_argument(
    "--pipeline_id",
    type=str,
    default="LWBenchhub-Autosim-G1OpenMicrowavePipeline-v0",
    help="Registered pipeline ID.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

import lw_benchhub.autosim  # noqa: F401 — triggers pipeline registration
from lw_benchhub.autosim.compat_autosim import patch_curobo_config, patch_env_extra_info, patch_reach_skill
from autosim import make_pipeline


def main():
    # EnvExtraInfo compatibility is needed by both X7S and G1 pipelines.
    patch_env_extra_info()

    # Reach/curobo runtime compatibility patches are only needed for G1 branch.
    if args_cli.pipeline_id in {
        "LWBenchhub-Autosim-G1OpenMicrowavePipeline-v0",
        "LWBenchhub-Autosim-G1LiftCubePipeline-v0",
    }:
        patch_reach_skill()
        patch_curobo_config()

    pipeline = make_pipeline(args_cli.pipeline_id)
    pipeline.run()


if __name__ == "__main__":
    main()

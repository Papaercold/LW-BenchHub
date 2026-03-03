"""Script to run autosim example."""


import argparse
from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="run autosim example pipeline.")
parser.add_argument("--pipeline_id", type=str, default=None, help="Name of the autosim pipeline.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app


import lw_benchhub.autosim

from autosim import make_pipeline


def main():
    pipeline = make_pipeline(args_cli.pipeline_id)
    pipeline.run()


if __name__ == '__main__':
    main()

from autosim import register_pipeline


register_pipeline(
    id="LWBenchhub-Autosim-CoffeeSetupMugPipeline-v0",
    entry_point=f"{__name__}.pipelines.coffee_setup_mug:CoffeeSetupMugPipeline",
    cfg_entry_point=f"{__name__}.pipelines.coffee_setup_mug:CoffeeSetupMugPipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-OpenFridgePipeline-v0",
    entry_point=f"{__name__}.pipelines.open_fridge:OpenFridgePipeline",
    cfg_entry_point=f"{__name__}.pipelines.open_fridge:OpenFridgePipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-CheesyBreadPipeline-v0",
    entry_point=f"{__name__}.pipelines.cheesy_bread:CheesyBreadPipeline",
    cfg_entry_point=f"{__name__}.pipelines.cheesy_bread:CheesyBreadPipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-CloseOvenPipeline-v0",
    entry_point=f"{__name__}.pipelines.close_oven:CloseOvenPipeline",
    cfg_entry_point=f"{__name__}.pipelines.close_oven:CloseOvenPipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-KettleBoilingPipeline-v0",
    entry_point=f"{__name__}.pipelines.kettle_boiling:KettleBoilingPipeline",
    cfg_entry_point=f"{__name__}.pipelines.kettle_boiling:KettleBoilingPipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-DessertUpgradePipeline-v0",
    entry_point=f"{__name__}.pipelines.dessert_upgrade:DessertUpgradePipeline",
    cfg_entry_point=f"{__name__}.pipelines.dessert_upgrade:DessertUpgradePipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-G1OpenMicrowavePipeline-v0",
    entry_point=f"{__name__}.pipelines.g1_open_microwave:G1OpenMicrowavePipeline",
    cfg_entry_point=f"{__name__}.pipelines.g1_open_microwave:G1OpenMicrowavePipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-G1OpenMicrowaveRightOnlyPipeline-v0",
    entry_point=f"{__name__}.pipelines.g1_open_microwave_right_only:G1OpenMicrowaveRightOnlyPipeline",
    cfg_entry_point=f"{__name__}.pipelines.g1_open_microwave_right_only:G1OpenMicrowaveRightOnlyPipelineCfg",
)

register_pipeline(
    id="LWBenchhub-Autosim-G1KettleBoilingPipeline-v0",
    entry_point=f"{__name__}.pipelines.kettle_boiling:KettleBoilingPipeline",
    cfg_entry_point=f"{__name__}.pipelines.kettle_boiling:KettleBoilingPipelineCfg",
    robot_profile="g1_loco_left",
)

register_pipeline(
    id="LWBenchhub-Autosim-G1CloseOvenPipeline-v0",
    entry_point=f"{__name__}.pipelines.close_oven:CloseOvenPipeline",
    cfg_entry_point=f"{__name__}.pipelines.close_oven:CloseOvenPipelineCfg",
    robot_profile="g1_loco_left",
)

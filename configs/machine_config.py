class MachineConfig:
    DOWNLOAD_MODEL_DIR = None
    CITYSCAPES_DIR = None
    GENERATED_DEPTH_DIR = None
    LOG_DIR = None
    AVAIL_MACHINES = ["ws", "bp1"]

    def __init__(self, machine):
        # if machine == "ws":
        #     MachineConfig.DOWNLOAD_MODEL_DIR = "/scratch/u5hv/shijie.u5hv/sde_seg/models/"
        #     MachineConfig.CITYSCAPES_DIR = "/scratch/u5hv/shijie.u5hv/sde_seg/raw/"
        #     MachineConfig.GENERATED_DEPTH_DIR = "/scratch/u5hv/shijie.u5hv/sde_seg/generated_depth/"
        #     MachineConfig.LOG_DIR = "/scratch/u5hv/shijie.u5hv/sde_seg/logs/"
        if machine in ["ws", "bp1"]:
            MachineConfig.DOWNLOAD_MODEL_DIR = "/user/work/ig23200/sde_seg/models/"
            MachineConfig.CITYSCAPES_DIR = "/user/work/ig23200/sde_seg/raw/"
            MachineConfig.GENERATED_DEPTH_DIR = "/user/work/ig23200/sde_seg/generated_depth/"
            MachineConfig.LOG_DIR = "/user/work/ig23200/sde_seg/logs/"
        else:
            raise NotImplementedError(machine)

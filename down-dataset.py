

# from roboflow import Roboflow
# rf = Roboflow(api_key="Vhb0bSrdSyCSj3zwPEVf")
# project = rf.workspace("gradproject-4qcgs").project("fire-smoke-lzyry")
# version = project.version(1)
# dataset = version.download("coco")



from roboflow import Roboflow
rf = Roboflow(api_key="Vhb0bSrdSyCSj3zwPEVf")
project = rf.workspace("new-workspace-p6pr5").project("fire-and-smoke-1tdrb")
version = project.version(6)
dataset = version.download("coco")
                
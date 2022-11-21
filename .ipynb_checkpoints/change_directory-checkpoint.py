import yaml
def change_UAV(source):
    with open('data/UAVDT.yaml') as f:
        Imagevid_yaml = yaml.load(f, Loader=yaml.FullLoader)
    Imagevid_yaml['path'] = source

    with open('data/UAVDT.yaml', 'w') as f:
        yaml.dump(Imagevid_yaml, f)

def change_VisDrone(source):
    with open('data/VisDroneVID.yaml') as f:
        Imagevid_yaml = yaml.load(f, Loader=yaml.FullLoader)
    Imagevid_yaml['path'] = source

    with open('data/VisDroneVID.yaml', 'w') as f:
        yaml.dump(Imagevid_yaml, f)

def change_AUAIR(source):
    with open('data/AUAIR.yaml') as f:
        Imagevid_yaml = yaml.load(f, Loader=yaml.FullLoader)
    Imagevid_yaml['path'] = source

    with open('data/AUAIR.yaml', 'w') as f:
        yaml.dump(Imagevid_yaml, f)

def change_ImageVID(source):
    with open('data/Imagevid.yaml') as f:
        Imagevid_yaml = yaml.load(f, Loader=yaml.FullLoader)
    Imagevid_yaml['path'] = source

    with open('data/Imagevid.yaml', 'w') as f:
        yaml.dump(Imagevid_yaml, f)

import yaml
def UAVDT_change_yaml(source):
    with open('data/UAVDT.yaml') as f:
        Imagevid_yaml = yaml.load(f, Loader=yaml.FullLoader)
    Imagevid_yaml['val'] = source+'/images'

    with open('data/UAVDT.yaml', 'w') as f:
        yaml.dump(Imagevid_yaml, f)

def VisDrone_change_yaml(source):
    with open('data/VisDroneVID.yaml') as f:
        Imagevid_yaml = yaml.load(f, Loader=yaml.FullLoader)
    Imagevid_yaml['val'] = source+'/images'

    with open('data/VisDroneVID.yaml', 'w') as f:
        yaml.dump(Imagevid_yaml, f)

def AUAIR_change_yaml(source):
    with open('data/AUAIR.yaml') as f:
        Imagevid_yaml = yaml.load(f, Loader=yaml.FullLoader)
    Imagevid_yaml['val'] = source+'/images'

    with open('data/AUAIR.yaml', 'w') as f:
        yaml.dump(Imagevid_yaml, f)

def ImageVID_change_yaml(source):
    with open('data/Imagevid.yaml') as f:
        Imagevid_yaml = yaml.load(f, Loader=yaml.FullLoader)
    Imagevid_yaml['val'] = source+'/images'

    with open('data/Imagevid.yaml', 'w') as f:
        yaml.dump(Imagevid_yaml, f)

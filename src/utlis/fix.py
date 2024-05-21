import os
import cv2
import pandas as pd
import yaml

DATASET_PATH = 'old_dataset'

def replay_and_save(filename: str, new_folder: str, cuts: list[tuple[int, int]]):    
    os.makedirs(os.path.join(new_folder, filename), exist_ok=True)
    
    images = list(filter(lambda x: 'jpg' in x, os.listdir(os.path.join(DATASET_PATH, filename))))
    images.sort(key=lambda x: int(x.removesuffix('.jpg')))
    
    for image in images:
        
        if any([int(image.removesuffix('.jpg')) >= start and int(image.removesuffix('.jpg')) <= stop for start, stop in cuts]):
            continue
        
        img = cv2.imread(os.path.join(DATASET_PATH, filename, image))
        
        # Save the image to the new folder
        cv2.imwrite(os.path.join(new_folder, filename, image), img)
        
    # Copy the laels csv to the new folder
    old = os.path.join(DATASET_PATH, f'{filename}.csv')
    new = os.path.join(new_folder, f'{filename}.csv')
    with open(old, 'r') as f:
        with open(new, 'w') as g:
            for line in f:
                frame = int(line.split(',')[0])
                if any([frame >= start and frame <= stop for start, stop in cuts]):
                    continue
                else:
                    g.write(line)
        
        
        
if __name__ == '__main__':
    
    
    # YAML read 
    with open(f"{DATASET_PATH}/config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)['files']
        except yaml.YAMLError as exc:
            print(exc)
    
    
    for record in config:
        filename = str(record['filename'])
        cuts = record['cuts']
        print(f"Filename: {filename}")
        print(f"Cuts: {cuts}")
        
        replay_and_save(filename, 'asdsa', cuts)
import os
import cv2

DATASET_PATH = 'dataset'
FOLDER = '1653043549.5187616'

def replay_frame_by_frame(cuts: list[tuple[int, int]]):
    '''
    Replay the frames of a run.
    '''
    images = list(filter(lambda x: 'jpg' in x, os.listdir(os.path.join(DATASET_PATH, FOLDER))))
    images.sort(key=lambda x: int(x.removesuffix('.jpg')))
    
    for image in images:
        
        # if int(image.removesuffix('.jpg')) < start_frame or int(image.removesuffix('.jpg')) > end_frame:
        if any([int(image.removesuffix('.jpg')) >= start and int(image.removesuffix('.jpg')) <= stop for start, stop in cuts]):
            continue
        
        img = cv2.imread(os.path.join(DATASET_PATH, FOLDER, image))
        
        # Make it bigger
        img = cv2.resize(img, (640, 480))
        
        cv2.imshow('Frame', img)
        
        # Print the frame number
        print(f'Frame: {image.removesuffix(".jpg")}')
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
def create_new_folder(path: str, name: str, cuts: list[tuple[int, int]]):
    '''
    Create a new folder in the dataset folder.
    '''
    os.makedirs(os.path.join(path, name))
    
    # Copy the images to the new folder but omit the ones outside the range
    images = list(filter(lambda x: 'jpg' in x, os.listdir(os.path.join(DATASET_PATH, FOLDER))))
    
    for image in images:
        if any([int(image.removesuffix('.jpg')) >= start and int(image.removesuffix('.jpg')) <= stop for start, stop in cuts]):
            continue
        
        img = cv2.imread(os.path.join(DATASET_PATH, FOLDER, image))
        cv2.imwrite(os.path.join(path, name, image), img)
        
    # Copy the labels
    old = os.path.join(DATASET_PATH, f'{name}.csv')
    new = os.path.join(path, f'{name}.csv')
    with open(old, 'r') as f:
        with open(new, 'w') as g:
            for line in f:
                frame = int(line.split(',')[0])
                if any([frame >= start and frame <= stop for start, stop in cuts]):
                    g.write(line)
        
    print(f'Folder {name} created.')
    
    with open(os.path.join(path, 'config.txt'), 'a+') as f:
        f.write(f'{name}, {cuts}\n')
    
if __name__ == '__main__':
    cuts = [(0, 55), (107, 1000)]
    replay_frame_by_frame(cuts)
    
    create_new_folder('new_data', FOLDER, cuts)
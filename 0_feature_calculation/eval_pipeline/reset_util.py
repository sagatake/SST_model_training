import os
import sys
import glob
import shutil
import pprint as pp


def reset_data_dir():

    root_dir = 'data'
    tgt_dir_prefix = 'data_'
    praat_output_dir = 'eval_pipeline/audio_face_util/temp_audio_out'
    openface_output_dir = 'eval_pipeline/audio_face_util/processed'
    lookup_file_path = 'data\\lookup.csv'
    feature_file_path = 'data\\features.csv'
    json_file_path = 'data\\eval_results.json'
    
    files_in_root = os.listdir(root_dir)
    data_XX_dirs = []
    for candidate_file in files_in_root:
        #print(candidate_file)
        candidate_path = os.path.join(root_dir, candidate_file)
        if (tgt_dir_prefix in candidate_file) and os.path.isdir(candidate_path):
            data_XX_dirs.append(candidate_path)
    data_XX_dirs.append(praat_output_dir)
    data_XX_dirs.append(openface_output_dir)
    #print('target dirs: ')
    #pp.pprint(data_XX_dirs)
    
    tgt_file_paths = []
    for data_XX_dir in data_XX_dirs:
        tgt_file_paths.extend(glob.glob('{}/*'.format(data_XX_dir), recursive=True))
    tgt_file_paths.append(lookup_file_path)
    tgt_file_paths.append(feature_file_path)
    tgt_file_paths.append(json_file_path)
    
    #print('target files: ')
    #pp.pprint(tgt_file_paths)
    
    """
    key = input('Are you sure to delete them? ([y]/n) :')
    if (key == 'n') or (key == 'N'):
        print('Aborted')
        sys.exit()
    """
    
    for data_dir in data_XX_dirs:
        
        shutil.rmtree(data_dir)
        os.mkdir(data_dir)
    
    for tgt_file in tgt_file_paths:
        try:
            os.remove(tgt_file)
        except:
            pass
        
    print('Reset successfully')
    
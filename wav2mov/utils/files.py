import os
import shutil

def move_dir(src,dest):
    os.makedirs(dest, exist_ok=True)
    print('src :',src)
    print('dest :',dest)
    shutil.move(src, dest)
    print(f'[SHUTIL]: Folder moved : src ({src}) dest ({dest})')
    
def mov_log_dir(options,config):
    src = os.path.dirname(config['log_fullpath'])
    if os.path.exists(src):
        dest = os.path.join(os.path.join(config['base_dir'],'logs'),
                            options.version)
        # os.makedirs(dest,exist_ok=True)
        move_dir(src,dest)
    
def mov_out_dir(options,config):
    if not options.test in ['y', 'yes']:
        return
    run = os.path.basename(options.model_path).strip('gen_').split('.')[0]
    version_dir = os.path.dirname(os.path.dirname(options.model_path))
    print('version dir',version_dir)
    version = os.path.basename(version_dir)
    if 'v' not in version:
        version = options.version
    src = os.path.join(config['out_dir'], run)
    if os.path.exists(src):
        dest = os.path.join(config['out_dir'], version)
        # os.makedirs(dest,exist_ok=True)
        move_dir(src, dest)
        
from .conversion import mol_to_smiles

def adock(receptor_input,
        smiles,
        ligand_name,
        center_x=7.750,
        center_y=-14.556,
        center_z=6.747,
        size_x=20,
        size_y=20,
        size_z=20,
        vina='qvina2',
        seed=None,
        cpu=1,
        lig_dir = './docking_score/ligand_files/',
        out_dir = './docking_score/output/',
        log_dir = './docking_score/log/',
        conf_dir = './docking_score/config/'):

    #Imports
    import os
    import subprocess
    import psutil
    import re

    timeout_duration = 1000

    #mkdir
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(conf_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(lig_dir, exist_ok=True)

    ligand = lig_dir + ligand_name + '.pdbqt'
    output = out_dir + ligand_name + '_out.pdbqt'
    config = conf_dir + ligand_name + '.conf'
    log = log_dir + ligand_name + '_log.txt'

    #Convert smiles
    if not os.path.isfile(ligand):
        with subprocess.Popen('obabel -:"' + smiles + '" -O ' + ligand + ' -h --gen3d' + ' > /dev/null 2>&1', \
                shell=True, start_new_session=True) as proc:
            try:
                proc.wait(timeout=timeout_duration)
            except subprocess.TimeoutExpired:
                p = psutil.Process(proc.pid)
                p.terminate()
    else:
        print(f'Ligand file: {ligand!r} already exists.')

    #Dock
    if os.path.isfile(receptor_input):
        if not os.path.isfile(output):
            #Create conf files
            conf = 'receptor = ' + receptor_input + '\n' +\
                    'ligand = ' + ligand + '\n' + \
                    'center_x = ' + str(center_x) + '\n' + \
                    'center_y = ' + str(center_y) + '\n' + \
                    'center_z = ' + str(center_z) + '\n' + \
                    'size_x = ' + str(size_x) + '\n' + \
                    'size_y = ' + str(size_y) + '\n' + \
                    'size_z = ' + str(size_z) + '\n' + \
                    'out = ' + output + '\n' + \
                    'cpu = ' + str(cpu)
            
            if seed is not None:
                conf += '\n' \
                    'seed = ' + str(seed)

            with open(config, 'w') as f:
                f.write(conf)
            with subprocess.Popen('' + vina + \
                    ' --config ' + config + \
                    ' --log ' + log + \
                    ' > /dev/null 2>&1', \
                    shell=True, start_new_session=True) as proc:
                try:
                    proc.wait(timeout=timeout_duration)
                except subprocess.TimeoutExpired:
                    p = psutil.Process(proc.pid)
                    p.terminate()
        result = 0
        try:
            score = float("inf")
            with open(output, 'r') as f:
                for line in f.readlines():
                    if "REMARK VINA RESULT" in line:
                        new_score = re.findall(r'([-+]?[0-9]*\.?[0-9]+)', line)[0]
                        score = min(score, float(new_score))
                result = score
        except FileNotFoundError:
            print('test--', ligand_name)
            result = 0
        
    else:
        print(f'Protein file: {receptor_input!r} not found!')
        result = 0

    return (result)


def calculateDockingScore(mol):
    protein_surface = './DATA/protein_files/6rqu.pdbqt'
    smi = mol_to_smiles(mol)
    ligand_name = smi.replace('(', '{').replace(')', '}')
    return adock(protein_surface, smi, ligand_name)

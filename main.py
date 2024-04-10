import sys, os
import subprocess
import argparse
import time

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
MODELS = {
        'vgg': {
            'dir':'./VGG_CIFAR',
            'exe':'combined_main.py',
            'custArgs':'',
        },
        'cyclegan': {
            'dir':'./CYCLE_GAN',
            'exe':'cyclegan_copy.py',
            'custArgs':'',
        },
        'lstm': {
            'dir':'./LSTM',
            'exe':'combined_main.py',
            'custArgs':'',
        },
        'test': {
            'dir':".",
            'exe':"test.py",
            'data':".",
            'custArgs':"-ctest -aabd".split(" ")
        }
       }
def _makeExec(model):
    return '{:s}'.format(model['exe'])
def runModel(m, numSteps, batchSize, mode, asy=0):
    # Run model with num_steps, batch_size
    command = ['python', _makeExec(m),
            '--batch_size', str(batchSize),
            '--num_steps', str(numSteps),
            '--job_type', mode,
            '--enable_perf_log']
    for a in m['custArgs']:
        command.append(a)
    with cd(m['dir']):
        print('Running command: ' + str(command))
        if asy:
            process = subprocess.Popen(command)
            return process
        else:
            subprocess.run(command)

def main(args):
    # Enable MPS if needed and store its previous state
    if args.MPS:
        pass
    # Run requested models and grab their throughput and latency
    models = []
    for m in args.models.split(","):
        if m in MODELS:
            models.append(MODELS[m])
        else:
            print("Model " + m + " not found! Exiting...")
            sys.exit(os.EX_USAGE)
    jobType = args.type.split(",")
    numSteps = args.num_steps.split(",")
    batchSize = args.batch_size.split(",")
    # If run model individually
    if args.colocate == 0:
        asy = 0
    # else colocate models
    else:
        asy = 1
    childProcess = []
    # for m in model: run it
    for i, m in enumerate(models):
        if jobType[i] == 'training':
            mode = 'training'
        else:
            mode = 'inference'
        process = runModel(m, numSteps=numSteps[i],
                batchSize=batchSize[i], asy=asy, mode=mode)
        childProcess.append(process)
        #time.sleep(20)
    if asy:
        for p in childProcess:
            p.wait()
    # Read readings from output file and store it in dict
    # for m in model: run it async
    # Generate a JSON file contianing measurements
    # Restore MPS state
if __name__=='__main__':
    parser = argparse.ArgumentParser(
            description='Run ML models on GPU and collect its performance data')

    parser.add_argument('--MPS', type=int, default=1,
            help='Whether to turn on or off MPS for the duration of\
            experiment(default on)')
    parser.add_argument('-c', '--colocate', type=int, default=1,
            help='Whether to run models listed by --model individually or\
            colocate them(default individually)')
    parser.add_argument('-m', '--models', type=str, default='vgg,cyclegan',
            help='Comma separated list of models to run. Supported: vgg,cyclegan.\
            Default:vgg,cyclegan')
#    parser.add_argument('-o', '--start_offset', type=int, default=30,
#            help='Start time(in second) from which to start taking\
#                    readings(default:30)')
#    parser.add_argument('-i', '--interval', type=int, default=100,
#            help='Interval(in seconds) for which to take readings(default:100)')
    parser.add_argument('-n', '--num_steps', type=str, default="1000,1000",
            help='Comma separated list of number of steps to run over which we want to take readings')
    parser.add_argument('-b', '--batch_size', type=str, default="20,20",
    help='Comma separated list of batch sizes to run models given by --models(default:4)')
    parser.add_argument('-t', '--type', type=str, default='training,training',
    choices=['training,inference','inference,inference','inference,training','training,training'], help='Comma separated list of mode in which\
    you want to run models specified by --models. Supported: training,\
    inference')
    args = parser.parse_args()
    main(args)

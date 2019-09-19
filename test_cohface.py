from joblib import Parallel, delayed
from vhr import VHR
import glob, os

COHFACE_DIR = '/home/vcuculo/Datasets/cohface/'
WINDOW = 10  # window size in seconds
STEP = 5  # step size in seconds
SHOW = 1  # show processing
METHOD = 2  # approach to use as in README
OUTDIR = "output/" + str(METHOD)  # output directory

if not os.path.exists(OUTDIR):
    os.makedirs(OUTDIR)

data = []

# create a test object for each video
for infile in glob.iglob(COHFACE_DIR + '/*/*/data.avi', recursive=True):
    outfile = OUTDIR + infile.replace(COHFACE_DIR, "").replace("/", "_").replace(".avi", ".csv")
    # print(outfile)
    data.append(VHR(infile, METHOD, WINDOW, STEP, outfile, SHOW))

num_cores = 1  # multiprocessing.cpu_count()-10
print("Processing %d files adopting %d cores..." % (len(data), num_cores))

Parallel(n_jobs=num_cores, verbose=100)(delayed(d.doCalc)() for d in data)

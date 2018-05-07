import multiprocessing

LOG_INTERVAL = 1
SAVE_INTERVAL = 2000


# Game Environment
FRAME_GAP = 8 # grame gap for one command


# Parameter
GAMMA = 0.99
GAE_GAMMA = GAMMA*0.95
ENTROPY_BETA = 0.01


# A3C
A3C_SAVE_FOLDER = "./a3c_cp/"
A3C_LOG_FOLDER = "./a3c_log/"
A3C_SYNC_INTERVAL = 50
A3C_N_WORKERS = 8 # max(1, int(multiprocessing.cpu_count()/2))
A3C_START_PORT = 2222 # start port for the local distribution system

# Fceux
import os
from distutils import spawn
CONFIG_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
FCEUX_SEARCH_PATH = os.pathsep.join([os.environ['PATH'], '/usr/games', '/usr/local/games'])
FCEUX_PATH = spawn.find_executable('fceux', FCEUX_SEARCH_PATH)
ROM_FILE = os.path.join(CONFIG_FILE_DIR, "SuperMarioBros/super-mario.nes")
PLUGIN_FILE = os.path.join(CONFIG_FILE_DIR, "SuperMarioBros/super-mario-bros.lua")
EMULATOR_LOST_DELAY = 5000 # ms

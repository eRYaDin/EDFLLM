from pathlib import Path
from eeglab.demo import make_demo

if __name__=='__main__':print('Demo dibuat:',make_demo(Path(__file__).resolve().parent/'demo'))

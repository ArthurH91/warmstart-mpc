import time
import numpy as np
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from planner import Planner
from scenes import Scene
from wrapper_panda import PandaWrapper

# Define custom progress bar
progress_bar = Progress(
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
)

# Creating the robot

T = 20 
robot_wrapper = PandaWrapper(capsule=True, auto_col=True)
rmodel, cmodel, vmodel = robot_wrapper()

name_scene = "wall"
scene = Scene(name_scene)
rmodel, cmodel, target, target2, q0 = scene.create_scene_from_urdf(rmodel, cmodel)
time_calc = []
results = []
n_samples = 10
# Use custom progress bar
with progress_bar as p:
    for i in p.track(range(n_samples)):
        # Do something here
        try:
            planner = Planner(rmodel, cmodel, scene, T)
            start = time.process_time()
            q_init, q_goal, X = planner.solve_and_optimize()
            t_solve = time.process_time() - start
            time_calc.append(t_solve)
            results.append([q_init, q_goal, X])
        except:
            print("failed solve, retrying")
            i -= 1
            continue
        
np.save(f'results_{name_scene}_{n_samples}.npy', np.array(results, dtype=object), allow_pickle=True)
np.save(f"time_result_{name_scene}_{n_samples}.npy", np.array(time_calc, dtype=object), allow_pickle=True)




import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.multiprocessing as mp

if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn")

import common_utils
from common_utils import Recorder
from common_utils import ibrl_utils as utils
from env.robosuite_wrapper import PixelRobosuite


class EvalProc:
    def __init__(
        self,
        seeds,
        process_id,
        env_params,
        terminal_queue: mp.Queue,
        record_dir=None,
        record_seeds=None,
    ):
        self.seeds = seeds
        self.process_id = process_id
        self.env_params: dict = env_params
        self.terminal_queue = terminal_queue
        self.send_queue = mp.Queue()
        self.recv_queue = mp.Queue()
        self.record_dir = record_dir
        # Set of seeds this worker should record; None or empty = no recording.
        self.record_seeds = set(record_seeds) if record_seeds else set()

    def start(self):
        env = PixelRobosuite(**self.env_params)
        recorder = (
            Recorder(self.record_dir)
            if (self.record_dir is not None and self.record_seeds)
            else None
        )

        results = {}
        for seed in self.seeds:
            np.random.seed(seed)
            obs, image_obs = env.reset()
            record_this = recorder is not None and seed in self.record_seeds
            # is_first_step lets the main process clear its per-worker action
            # queue at episode boundaries (otherwise it'd send stale actions
            # from the previous episode's chunk).
            is_first = True
            success = False
            while not env.terminal:
                if record_this:
                    recorder.add(image_obs)
                # NOTE: obs["obs"] should be a cpu tensor because it
                # is more complicated to move cuda tensors around.
                self.send_queue.put((self.process_id, obs, is_first))
                is_first = False
                action = self.recv_queue.get()
                obs, _, _, success, image_obs = env.step(action)

            results[seed] = float(success)
            if record_this:
                recorder.save(f"episode_seed{seed}")

        self.terminal_queue.put((self.process_id, results))
        return


def _policy_act_chunk(agent, batch_obs):
    """Call agent.act_chunk if available; else wrap agent.act as length-1 chunk.

    Returns tensor (B, T, action_dim) on CPU.
    """
    if hasattr(agent, "act_chunk"):
        return agent.act_chunk(batch_obs, eval_mode=True)
    actions = agent.act(batch_obs, eval_mode=True)
    return actions.unsqueeze(1)  # (B, D) -> (B, 1, D)


def run_eval(
    env_params, agent, num_game, num_proc, seed, verbose=True,
    record_dir=None, record_every=0,
) -> list[float]:
    assert num_game % num_proc == 0
    env_params["device"] = "cpu"  # avoid sending cuda across processes

    game_per_proc = num_game // num_proc
    terminal_queue = mp.Queue()

    # Decide which seeds to record. record_every > 0 picks every Nth episode by
    # global position within this eval run (offset from `seed`).
    if record_dir is not None and record_every > 0:
        record_seeds_all = {
            seed + i for i in range(num_game) if i % record_every == 0
        }
    else:
        record_seeds_all = set()

    eval_procs = []
    for i in range(num_proc):
        seeds = list(range(seed + i * game_per_proc, seed + (i + 1) * game_per_proc))
        worker_record_seeds = record_seeds_all.intersection(seeds)
        eval_procs.append(EvalProc(
            seeds, i, env_params, terminal_queue,
            record_dir=record_dir,
            record_seeds=worker_record_seeds,
        ))

    put_queues = {i: proc.recv_queue for i, proc in enumerate(eval_procs)}
    get_queues = {i: proc.send_queue for i, proc in enumerate(eval_procs)}

    processes = {i: mp.Process(target=proc.start) for i, proc in enumerate(eval_procs)}
    for _, p in processes.items():
        p.start()

    # Per-worker action queues. Refilled lazily from policy.act_chunk; cleared
    # when worker signals is_first=True (start of a new episode). Independent
    # of policy class — non-chunked policies just get length-1 chunks.
    action_queues: dict = {i: deque() for i in range(num_proc)}

    t = time.time()
    results = {}
    with torch.no_grad(), utils.eval_mode(agent):
        while len(processes) > 0:
            while not terminal_queue.empty():
                term_idx, proc_results = terminal_queue.get()
                results.update(proc_results)
                processes[term_idx].join()
                processes.pop(term_idx)
                get_queues.pop(term_idx)
                put_queues.pop(term_idx)
                action_queues.pop(term_idx, None)

            obses = defaultdict(list)
            idxs = []
            for proc_idx, get_queue in get_queues.items():
                if get_queue.empty():
                    continue
                proc_id, obs_dict, is_first = get_queue.get()
                if is_first:
                    # New episode: drop any stale actions from the previous one
                    action_queues[proc_id].clear()
                    if hasattr(agent, "reset_prior"):
                        agent.reset_prior([proc_id])      # the IBRL prior's chunk queue (prior_chunk > 1)
                idxs.append(proc_id)
                for k, v in obs_dict.items():
                    obses[k].append(v)

            if len(obses) == 0:
                continue

            # Only ask the policy to predict for idxs whose action queue is
            # empty; reuse already-queued actions for the rest. This is what
            # makes chunking actually cheap (we call act() once per Ta env steps
            # in the limit, instead of every step).
            need_pred_pos = [pos for pos, idx in enumerate(idxs) if not action_queues[idx]]

            if need_pred_pos:
                pred_obs = {
                    k: torch.stack([v[pos] for pos in need_pred_pos]).cuda()
                    for k, v in obses.items()
                }
                if hasattr(agent, "prior_keys"):         # which worker each row is (IBRL prior_chunk queues)
                    agent.prior_keys = [idxs[pos] for pos in need_pred_pos]
                try:
                    chunks = _policy_act_chunk(agent, pred_obs)  # (B_need, T, D)
                finally:
                    if hasattr(agent, "prior_keys"):
                        agent.prior_keys = None
                # chunks is on CPU (per act_chunk contract); push into queues
                for j, pos in enumerate(need_pred_pos):
                    idx = idxs[pos]
                    for a in chunks[j]:
                        action_queues[idx].append(a)

            # Pop one action per worker and send back
            for pos, idx in enumerate(idxs):
                action = action_queues[idx].popleft()
                put_queues[idx].put(action)

    if verbose:
        print(f"total time {time.time() - t:.2f}")
        for s in sorted(list(results.keys())):
            print(f"seed {s}: score: {float(results[s])}")
        print(common_utils.wrap_ruler(""))

    scores = []
    for s in sorted(list(results.keys())):
        scores.append(results[s])
    return scores

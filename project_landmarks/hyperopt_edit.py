from hyperopt import fmin, tpe, STATUS_OK, STATUS_FAIL, Trials, hp, space_eval, FMinIter
from hyperopt.base import JOB_STATES, JOB_STATE_NEW, JOB_STATE_DONE, JOB_STATE_RUNNING, JOB_STATE_ERROR, JOB_VALID_STATES

import numpy as np

# JOB_STATE_NEW = 0
# JOB_STATE_RUNNING = 1
# JOB_STATE_DONE = 2
# JOB_STATE_ERROR = 3
# JOB_STATES = [
#     JOB_STATE_NEW,
#     JOB_STATE_RUNNING,
#     JOB_STATE_DONE,
#     JOB_STATE_ERROR,
#     JOB_STATE_CANCEL,
# ]
# JOB_VALID_STATES = {JOB_STATE_NEW, JOB_STATE_RUNNING, JOB_STATE_DONE}

### Redefine fmin (changed n_queued to len(trials)
def run(self, N, block_until_done=True):
    """
    Run `self.algo` iteratively (use existing `self.trials` to produce the new
    ones), update, and repeat
    block_until_done  means that the process blocks until ALL jobs in
    trials are not in running or new state

    """
    trials = self.trials
    algo = self.algo
    n_queued = 0

    def get_queue_len():
        return self.trials.count_by_state_unsynced(JOB_STATE_NEW)

    def get_n_done():
        return self.trials.count_by_state_unsynced(JOB_STATE_DONE)

    def get_n_unfinished():
        unfinished_states = [JOB_STATE_NEW, JOB_STATE_RUNNING]
        return self.trials.count_by_state_unsynced(unfinished_states)

    stopped = False
    initial_n_done = get_n_done()
    with self.progress_callback(
        initial=initial_n_done, total=self.max_evals
    ) as progress_ctx:
        all_trials_complete = False
        best_loss = float("inf")
        while (
            # more run to Q     || ( block_flag & trials not done )
            (n_queued < N or (block_until_done and not all_trials_complete))
            # no timeout        || < current last time
            and (self.timeout is None or (timer() - self.start_time) < self.timeout)
            # no loss_threshold || < current best_loss
            and (self.loss_threshold is None or best_loss >= self.loss_threshold)
        ):
            qlen = get_queue_len()
            while (
                qlen < self.max_queue_len and n_queued < N and not self.is_cancelled
            ):  
                n_to_enqueue = min(self.max_queue_len - qlen, N - n_queued)
                # get ids for next trials to enqueue
                new_ids = trials.new_trial_ids(n_to_enqueue)
                self.trials.refresh()
                # Based on existing trials and the domain, use `algo` to probe in
                # new hp points. Save the results of those inspections into
                # `new_trials`. This is the core of `run`, all the rest is just
                # processes orchestration
                new_trials = algo(
                    new_ids, self.domain, trials, self.rstate.integers(2**31 - 1)
                )
                assert len(new_ids) >= len(new_trials)

                if len(new_trials):
                    self.trials.insert_trial_docs(new_trials)
                    self.trials.refresh()
                    # n_queued += len(new_trials)
                    # n_queued = len(self.trials)
                    n_queued = get_n_done()
                    qlen = get_queue_len()
                else:
                    stopped = True
                    break

            if self.is_cancelled:
                break

            if self.asynchronous:
                # -- wait for workers to fill in the trials
                time.sleep(self.poll_interval_secs)
            else:
                # -- loop over trials and do the jobs directly
                self.serial_evaluate()

            self.trials.refresh()
            if self.trials_save_file != "":
                pickler.dump(self.trials, open(self.trials_save_file, "wb"))
            if self.early_stop_fn is not None:
                stop, kwargs = self.early_stop_fn(
                    self.trials, *self.early_stop_args
                )
                self.early_stop_args = kwargs
                if stop:
                    logger.info(
                        "Early stop triggered. Stopping iterations as condition is reach."
                    )
                    stopped = True
            # update progress bar with the min loss among trials with status ok
            losses = [
                loss
                for loss in self.trials.losses()
                if loss is not None and not np.isnan(loss)
            ]
            if losses:
                best_loss = min(losses)
                progress_ctx.postfix = "best loss: " + str(best_loss)

            n_unfinished = get_n_unfinished()
            if n_unfinished == 0:
                all_trials_complete = True

            n_done = get_n_done()
            n_done_this_iteration = n_done - initial_n_done
            if n_done_this_iteration > 0:
                progress_ctx.update(n_done_this_iteration)
            initial_n_done = n_done

            if stopped:
                break

### Trials: changed refresh to update trials (do not count SATUS_FAIL)

def refresh(self):
    # In MongoTrials, this method fetches from database
    if self._exp_key is None:
        statuses = [x['result']['status'] for x in self._dynamic_trials]
        for i, status in enumerate(statuses):
            if status == STATUS_FAIL:
                self._dynamic_trials[i]['state'] = JOB_STATE_ERROR
        self._trials = [
            tt for tt in self._dynamic_trials if tt["state"] in JOB_VALID_STATES
        ]
    else:
        self._trials = [
            tt
            for tt in self._dynamic_trials
            if (tt["state"] in JOB_VALID_STATES and tt["exp_key"] == self._exp_key)
        ]
    self._ids.update([tt["tid"] for tt in self._trials])
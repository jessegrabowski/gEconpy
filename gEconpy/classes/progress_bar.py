import time

import numpy as np


class ProgressBar:
    """
    A utility class for displaying a progress bar in the command line.
    """

    def __init__(self, total, verb="", start_iters=0, bar_length=50):
        """
        Initialize a ProgressBar instance.

        Parameters
        ----------
        total : int
            Total number of iterations.
        verb : str, optional
            String to be displayed before the progress bar. The default is ''.
        start_iters : int, optional
            Number of iterations that have already been completed. The default is 0.
        bar_length : int, optional
            Length of the progress bar in characters. The default is 50.
        """

        self.total = total
        self.verb = verb
        self.start_iters = start_iters
        self.bar_length = bar_length

        self.start_time = None
        self.mean_time = 0
        self.n_iters = 0

        self.init_time = time.time()
        self.last_print_time = 0

    def start(self):
        """Start tracking time for a loop iteration."""
        self.n_iters += 1
        self.start_time = time.time()

    def stop(self):
        """Stop tracking time for a loop iteration and update mean time using the Robins-Monroe algorithm."""

        alpha = 1 / (self.n_iters + 1)
        elapsed = time.time() - self.start_time
        self.mean_time = alpha * elapsed + (1 - alpha) * self.mean_time

        if (time.time() - self.last_print_time > 0.25) or (self.n_iters == self.total):
            self.print_progress()

    @staticmethod
    def _time_to_string(timestamp):
        """
        Convert a time in seconds to a string in the format "mm:ss".

        Parameters
        ----------
        timestamp : float
            Time in seconds.

        Returns
        -------
        Tuple[str, str]
            Tuple of strings in the format "mm:ss".
        """

        minutes, seconds = np.divmod(timestamp, 60)
        minutes = int(minutes)
        minutes = "0" * (2 - len(str(minutes))) + str(minutes)
        seconds = int(seconds)
        seconds = "0" * (2 - len(str(seconds))) + str(seconds)

        return minutes, seconds

    def print_progress(self):
        """
        Print the current progress and remaining time to completion.
        """

        remaining = self.mean_time * (self.total - self.n_iters)
        elapsed = time.time() - self.init_time

        remain_min, remain_sec = self._time_to_string(remaining)
        elapse_min, elapse_sec = self._time_to_string(elapsed)

        iter_per_sec = self.n_iters / (elapsed + 1e-8)

        n_digits = len(str(self.total))

        total_iters = self.start_iters + self.n_iters
        pct_complete = int(total_iters / self.total * self.bar_length)

        bar = f"{self.verb} {total_iters:<{n_digits}} / {self.total} ["
        bar = bar + "=" * pct_complete + " " * (self.bar_length - pct_complete) + "]"

        time_info = f"elapsed: {elapse_min}:{elapse_sec}, "
        time_info += f"remaining: {remain_min}:{remain_sec}, "

        if iter_per_sec < 1:
            time_info += f"{1 / iter_per_sec:0.2f}sec/iter"
        else:
            time_info += f"{iter_per_sec:0.2f}iter/sec"

        complete = self.n_iters == self.total
        print(bar, time_info, end="\n" if complete else "\r")
        self.last_print_time = time.time()

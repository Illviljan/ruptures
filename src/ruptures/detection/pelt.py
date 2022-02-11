r"""Pelt"""
from math import floor

import numpy as np

from ruptures.costs import cost_factory
from ruptures.base import BaseCost, BaseEstimator
from ruptures.exceptions import BadSegmentationParameters
from ruptures.utils import sanity_check


class Pelt(BaseEstimator):

    """Penalized change point detection.

    For a given model and penalty level, computes the segmentation which
    minimizes the constrained sum of approximation errors.
    """

    def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
        """Initialize a Pelt instance.

        Args:
            model (str, optional): segment model, ["l1", "l2", "rbf"]. Not used if ``'custom_cost'`` is not None.
            custom_cost (BaseCost, optional): custom cost function. Defaults to None.
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the cost instance.
        """
        if custom_cost is not None and isinstance(custom_cost, BaseCost):
            self.cost = custom_cost
        else:
            if params is None:
                self.cost = cost_factory(model=model)
            else:
                self.cost = cost_factory(model=model, **params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_samples = None

    def _seg(self, pen):
        """Computes the segmentation for a given penalty using PELT (or a list
        of penalties).

        Args:
            penalty (float): penalty value

        Returns:
            dict: partition dict {(start, end): cost value,...}
        """
        from pprint import pprint
        n_samples = self.n_samples
        jump = self.jump
        min_size  = self.min_size
        signal = self.cost.signal
        cost_error = self.cost.error
        
        # initialization
        # partitions[t] contains the optimal partition of signal[0:t]
        partitions = dict()  # this dict will be recursively filled
        partitions[0] = {(0, 0): 0}
        admissible = []

        # Recursion
        # ind = [k for k in range(0, n_samples, jump) if k >= min_size]
        # ind += [n_samples]
        
        ind = np.arange(np.ceil(min_size/jump)*jump, np.ceil(n_samples/jump)*jump + 1, jump, dtype=np.int_)
        ind[-1] = n_samples
        
        admissible_unmasked = np.arange(0, np.ceil(n_samples/jump)*jump, jump, dtype=np.int_)
        admissible_ind = np.arange(admissible_unmasked.size, dtype=np.int_)
        
        subproblems_unmasked = np.empty_like(admissible_unmasked, dtype=np.float_)
        subproblems_unmasked[:] = np.inf
        partitions_unmasked = np.zeros(n_samples+1, dtype=np.float_)
        mask = np.zeros_like(admissible_unmasked, dtype=np.bool_)
        # # Handle the inner loop try, except:
        # not_under_min_size = (0 < admissible_unmasked) & (admissible_unmasked < min_size)
        # mask[not_under_min_size] = True # Becomes False in loop
        bkp_old = 1
        for j, bkp in enumerate(ind):
            # # adding a point to the admissible set from the previous loop.
            new_adm_pt = floor((bkp - min_size) / jump)
            new_adm_pt *= jump
            admissible.append(new_adm_pt)
            
            subproblems = list()
            if not (0 < admissible_unmasked[j] < min_size): # Handle the inner loop try, except
                mask[j] = True
            admissible_masked = admissible_unmasked[mask]
            print(j, bkp)
            print(admissible)
            print(admissible_masked.tolist())
            assert admissible == (admissible_masked.tolist() + ([admissible_unmasked[j]] if (0 < admissible_unmasked[j] < min_size) else []))
            
            subproblems_masked = subproblems_unmasked[mask]
            subproblems_masked2 = subproblems_unmasked[mask]
            for jj, t in enumerate(admissible):
                # left partition
                try:
                    tmp_partition = partitions[t].copy()
                    tmp_partitions_ = partitions_unmasked[t]
                except KeyError:  # no partition of 0:t exists points before min_size
                    continue
                # we update with the right partition
                tmp_partition.update({(t, bkp): cost_error(t, bkp) + pen})
                subproblems.append(tmp_partition)
                subproblems_masked[jj] = tmp_partitions_ + cost_error(t, bkp) + pen
                assert np.isclose(sum(tmp_partition.values()), subproblems_masked[jj])
            
            subproblems_masked2[:] = partitions_unmasked[admissible_masked] + pen
            subproblems_masked2[:] += np.array([cost_error(t, bkp) for t in admissible_masked])
            assert all(np.isclose(subproblems_masked, subproblems_masked2))
            
            # finding the optimal partition
            print(f"{bkp=}")
            print("subproblems:")
            pprint(subproblems)
            partitions[bkp] = min(subproblems, key=lambda d: sum(d.values()))
            v_min = np.inf
            i_min = -1
            for i, d in enumerate(subproblems):
                val = sum(d.values())
                if val < v_min:
                    v_min = val
                    i_min = i
                     
            
            print("partitions[bkp]:")
            pprint(partitions[bkp])
            pprint(sum(partitions[bkp].values()))
            print("for loop min:")
            pprint(subproblems[i_min])
            assert subproblems[i_min] == partitions[bkp]
 
            assert np.isclose(np.min(subproblems_masked), np.min(subproblems_masked2))
            partitions_unmasked[bkp_old:bkp+1] = np.min(subproblems_masked)
            # partitions_unmasked[bkp] = np.min(subproblems_masked)
            pprint(f"{partitions_unmasked[bkp] = }")  
            assert np.isclose(partitions_unmasked[bkp], sum(partitions[bkp].values()))
            
            # trimming the admissible set
            admissible_ = []
            # subproblems can be shorter than admissible:
            for t, partition in zip(admissible, subproblems):
                p_t = sum(partition.values())
                p_bkp = sum(partitions[bkp].values()) + pen
                print(t, p_t, p_bkp)
                if p_t <= p_bkp:
                    admissible_.append(t)
                #     mask[t] = True
                # else:
                #     mask[t] = False
            # When subproblem is smaller than admission:
            # mask[admissible[len(subproblems):]] = False 
            admissible = admissible_
            
            p_t = subproblems_masked
            p_bkp = partitions_unmasked[bkp] + pen
            mask[admissible_masked] = p_t <= p_bkp
            bkp_old = bkp

        best_partition = partitions[n_samples]
        del best_partition[(0, 0)]
        return best_partition

    def fit(self, signal) -> "Pelt":
        """Set params.

        Args:
            signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update params
        self.cost.fit(signal)
        if signal.ndim == 1:
            (n_samples,) = signal.shape
        else:
            n_samples, _ = signal.shape
        self.n_samples = n_samples
        return self

    def predict(self, pen):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to [`fit()`][ruptures.detection.pelt.Pelt.fit].

        Args:
            pen (float): penalty value (>0)

        Raises:
            BadSegmentationParameters: in case of impossible segmentation
                configuration

        Returns:
            list: sorted list of breakpoints
        """
        # raise an exception in case of impossible segmentation configuration
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=0,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters

        partition = self._seg(pen)
        bkps = sorted(e for s, e in partition.keys())
        return bkps

    def fit_predict(self, signal, pen):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            pen (float): penalty value (>0)

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(pen)


# %% Original
# r"""Pelt"""
# from math import floor

# from ruptures.costs import cost_factory
# from ruptures.base import BaseCost, BaseEstimator
# from ruptures.exceptions import BadSegmentationParameters
# from ruptures.utils import sanity_check


# class Pelt(BaseEstimator):

#     """Penalized change point detection.

#     For a given model and penalty level, computes the segmentation which
#     minimizes the constrained sum of approximation errors.
#     """

#     def __init__(self, model="l2", custom_cost=None, min_size=2, jump=5, params=None):
#         """Initialize a Pelt instance.

#         Args:
#             model (str, optional): segment model, ["l1", "l2", "rbf"]. Not used if ``'custom_cost'`` is not None.
#             custom_cost (BaseCost, optional): custom cost function. Defaults to None.
#             min_size (int, optional): minimum segment length.
#             jump (int, optional): subsample (one every *jump* points).
#             params (dict, optional): a dictionary of parameters for the cost instance.
#         """
#         if custom_cost is not None and isinstance(custom_cost, BaseCost):
#             self.cost = custom_cost
#         else:
#             if params is None:
#                 self.cost = cost_factory(model=model)
#             else:
#                 self.cost = cost_factory(model=model, **params)
#         self.min_size = max(min_size, self.cost.min_size)
#         self.jump = jump
#         self.n_samples = None

#     def _seg(self, pen):
#         """Computes the segmentation for a given penalty using PELT (or a list
#         of penalties).

#         Args:
#             penalty (float): penalty value

#         Returns:
#             dict: partition dict {(start, end): cost value,...}
#         """

#         # initialization
#         # partitions[t] contains the optimal partition of signal[0:t]
#         partitions = dict()  # this dict will be recursively filled
#         partitions[0] = {(0, 0): 0}
#         admissible = []

#         # Recursion
#         ind = [k for k in range(0, self.n_samples, self.jump) if k >= self.min_size]
#         ind += [self.n_samples]
#         for bkp in ind:
#             # adding a point to the admissible set from the previous loop.
#             new_adm_pt = floor((bkp - self.min_size) / self.jump)
#             new_adm_pt *= self.jump
#             admissible.append(new_adm_pt)

#             subproblems = list()
#             for t in admissible:
#                 # left partition
#                 try:
#                     tmp_partition = partitions[t].copy()
#                 except KeyError:  # no partition of 0:t exists
#                     continue
#                 # we update with the right partition
#                 tmp_partition.update({(t, bkp): self.cost.error(t, bkp) + pen})
#                 subproblems.append(tmp_partition)

#             # finding the optimal partition
#             partitions[bkp] = min(subproblems, key=lambda d: sum(d.values()))
#             # trimming the admissible set
#             admissible = [
#                 t
#                 for t, partition in zip(admissible, subproblems)
#                 if sum(partition.values()) <= sum(partitions[bkp].values()) + pen
#             ]

#         best_partition = partitions[self.n_samples]
#         del best_partition[(0, 0)]
#         return best_partition

#     def fit(self, signal) -> "Pelt":
#         """Set params.

#         Args:
#             signal (array): signal to segment. Shape (n_samples, n_features) or (n_samples,).

#         Returns:
#             self
#         """
#         # update params
#         self.cost.fit(signal)
#         if signal.ndim == 1:
#             (n_samples,) = signal.shape
#         else:
#             n_samples, _ = signal.shape
#         self.n_samples = n_samples
#         return self

#     def predict(self, pen):
#         """Return the optimal breakpoints.

#         Must be called after the fit method. The breakpoints are associated with the signal passed
#         to [`fit()`][ruptures.detection.pelt.Pelt.fit].

#         Args:
#             pen (float): penalty value (>0)

#         Raises:
#             BadSegmentationParameters: in case of impossible segmentation
#                 configuration

#         Returns:
#             list: sorted list of breakpoints
#         """
#         # raise an exception in case of impossible segmentation configuration
#         if not sanity_check(
#             n_samples=self.cost.signal.shape[0],
#             n_bkps=0,
#             jump=self.jump,
#             min_size=self.min_size,
#         ):
#             raise BadSegmentationParameters

#         partition = self._seg(pen)
#         bkps = sorted(e for s, e in partition.keys())
#         return bkps

#     def fit_predict(self, signal, pen):
#         """Fit to the signal and return the optimal breakpoints.

#         Helper method to call fit and predict once

#         Args:
#             signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
#             pen (float): penalty value (>0)

#         Returns:
#             list: sorted list of breakpoints
#         """
#         self.fit(signal)
#         return self.predict(pen)
